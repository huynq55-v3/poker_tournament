use crate::deep_cfr::advantage_memory::{AdvantageMemory, AdvantageSample, SampleBatch};
use crate::deep_cfr::policy::DeepCFRPolicy;
use crate::deep_cfr::strategy_memory::{StrategyMemory, StrategySample};
use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::neural_network::{LossType, MLP};
use crate::sampler::TournamentStackSampler;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Số node của traverser được rẽ nhánh đầy đủ trên một đường đi
const MAX_BRANCH_DEPTH: usize = 4;
const ICM_SCALE: f32 = 10.0;

fn payout_structure(num_players: usize) -> Vec<f32> {
    match num_players {
        0..=2 => vec![1.0],
        3 => vec![0.65, 0.35],
        4..=5 => vec![0.5, 0.3, 0.2],
        _ => vec![0.4, 0.3, 0.2, 0.1],
    }
}

/// ICM Malmuth-Harville
pub fn icm_equity(stacks: &[f32], payouts: &[f32]) -> Vec<f32> {
    fn rec(stacks: &[f32], remaining: u32, place: usize, prob: f32, payouts: &[f32], eq: &mut [f32]) {
        if place >= payouts.len() {
            return;
        }
        let mut total = 0.0f32;
        for (i, &s) in stacks.iter().enumerate() {
            if remaining & (1 << i) != 0 {
                total += s;
            }
        }
        if total <= 0.0 {
            return;
        }
        for i in 0..stacks.len() {
            if remaining & (1 << i) != 0 && stacks[i] > 0.0 {
                let p = prob * stacks[i] / total;
                eq[i] += p * payouts[place];
                rec(stacks, remaining & !(1 << i), place + 1, p, payouts, eq);
            }
        }
    }
    let n = stacks.len();
    let mut eq = vec![0.0f32; n];
    let all = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
    rec(stacks, all, 0, 1.0, payouts, &mut eq);
    eq
}

/// Hàm thưởng cho 1 traverser trong 1 ván. `before` (equity ICM ban đầu) tính 1 lần.
struct Payoff {
    traverser: usize,
    start_stack: f32,
    payouts: Vec<f32>,
    before: f32,
    use_icm: bool,
    bb: f32,
}

impl Payoff {
    fn new(start: &[f32], traverser: usize, use_icm: bool, bb: f32) -> Self {
        let payouts = payout_structure(start.len());
        let before = if use_icm { icm_equity(start, &payouts)[traverser] } else { 0.0 };
        Payoff { traverser, start_stack: start[traverser], payouts, before, use_icm, bb }
    }

    fn eval(&self, env: &Environment) -> f32 {
        if self.use_icm {
            let end: Vec<f32> = env.state.players.iter().map(|p| p.stack as f32).collect();
            let after = icm_equity(&end, &self.payouts)[self.traverser];
            (after - self.before) * 100.0 / ICM_SCALE
        } else {
            // chip-EV, đơn vị BB / 10 (cùng bậc độ lớn với ICM)
            let end = env.state.players[self.traverser].stack as f32;
            (end - self.start_stack) / self.bb / 10.0
        }
    }
}

/// Ngữ cảnh của 1 luồng traversal: mạng chỉ đọc + buffer riêng
struct Ctx<'a> {
    net: &'a MLP,
    adv_out: Vec<AdvantageSample>,
    strat_out: Vec<StrategySample>,
    weight: f32,
}

fn sample_action<R: Rng>(probs: &[f32; NUM_ACTIONS], mask: &ActionMask, rng: &mut R) -> Action {
    if let Ok(dist) = WeightedIndex::new(probs.iter()) {
        let a_idx = dist.sample(rng);
        if mask.is_index_valid(a_idx) {
            if let Some(a) = Action::from_index(a_idx) {
                return a;
            }
        }
    }
    mask.valid_actions().first().copied().unwrap_or(Action::Fold)
}

fn step_and_continue<R: Rng>(
    env: &mut Environment,
    action: Action,
    ctx: &mut Ctx,
    pay: &Payoff,
    rng: &mut R,
    depth: usize,
) -> f32 {
    match env.step(action) {
        Ok(false) => traverse(env, ctx, pay, rng, depth),
        _ => pay.eval(env),
    }
}

/// External-sampling MCCFR
fn traverse<R: Rng>(env: &mut Environment, ctx: &mut Ctx, pay: &Payoff, rng: &mut R, depth: usize) -> f32 {
    if env.state.street == Street::Showdown {
        return pay.eval(env);
    }
    if env.state.players.iter().filter(|p| p.is_in_hand()).count() <= 1 {
        return pay.eval(env);
    }

    let curr = env.state.current_player_idx;
    let mask = env.get_action_mask();
    if mask.0 == 0 {
        return pay.eval(env);
    }

    let features = env.state.encode_features(curr);
    let raw = ctx.net.forward(&features);
    let strat = DeepCFRPolicy::compute_regret_matching_probs(&raw, &mask);

    if curr == pay.traverser {
        if depth >= MAX_BRANCH_DEPTH {
            let a = sample_action(&strat, &mask, rng);
            return step_and_continue(env, a, ctx, pay, rng, depth);
        }

        let valid = mask.valid_actions();
        let mut values = [0.0f32; NUM_ACTIONS];
        let mut expected = 0.0f32;
        for &action in &valid {
            let i = action.to_index();
            let mut branch = env.clone();
            let v = step_and_continue(&mut branch, action, ctx, pay, rng, depth + 1);
            values[i] = v;
            expected += strat[i] * v;
        }

        let mut regrets = vec![0.0f32; NUM_ACTIONS];
        for &action in &valid {
            let i = action.to_index();
            regrets[i] = values[i] - expected;
        }
        ctx.adv_out.push(AdvantageSample {
            features,
            regrets,
            mask: mask.to_bool_array(),
            iteration_weight: ctx.weight,
        });
        expected
    } else {
        ctx.strat_out.push(StrategySample {
            features,
            action_probs: strat.to_vec(),
            mask: mask.to_bool_array(),
            iteration_weight: ctx.weight,
        });
        let a = sample_action(&strat, &mask, rng);
        step_and_continue(env, a, ctx, pay, rng, depth)
    }
}

pub struct DeepCFRTrainer {
    pub advantage_net: Arc<RwLock<MLP>>,
    pub strategy_net: Arc<RwLock<MLP>>,
    pub adv_memory: AdvantageMemory,
    pub strategy_memory: StrategyMemory,
    pub strategy_trained: bool,
    pub iteration: usize,
    pub lr: f32,
    pub weight_decay: f32,
    /// false = chip-EV (giai đoạn đầu), true = ICM
    pub use_icm: bool,
    /// Some(n) = luôn train bàn n người; None = ngẫu nhiên 2..=8
    pub fixed_players: Option<usize>,
}

impl DeepCFRTrainer {

    /// Nạp 2 mạng từ file để train tiếp. Replay buffer vẫn rỗng, optimizer được reset.
    /// Trả về Ok(true) nếu nạp được cả strategy net.
    pub fn load_models(&mut self, adv_path: &str, strat_path: &str) -> Result<bool, String> {
        let expected = self.advantage_net.read().unwrap().layer_dims();

        let mut adv = MLP::load_from_file(adv_path).map_err(|e| format!("{}: {}", adv_path, e))?;
        if adv.layer_dims() != expected {
            return Err(format!(
                "kiến trúc không khớp: file {:?} vs trainer {:?}",
                adv.layer_dims(),
                expected
            ));
        }
        adv.reset_optimizer();
        *self.advantage_net.write().unwrap() = adv;

        match MLP::load_from_file(strat_path) {
            Ok(mut s) if s.layer_dims() == expected => {
                s.reset_optimizer();
                *self.strategy_net.write().unwrap() = s;
                self.strategy_trained = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    
    pub fn new(input_dim: usize, hidden_dims: &[usize], capacity: usize) -> Self {
        let mut dims = Vec::with_capacity(hidden_dims.len() + 2);
        dims.push(input_dim);
        dims.extend_from_slice(hidden_dims);
        dims.push(NUM_ACTIONS);

        Self {
            advantage_net: Arc::new(RwLock::new(MLP::new(&dims))),
            strategy_net: Arc::new(RwLock::new(MLP::new(&dims))),
            adv_memory: AdvantageMemory::new(capacity),
            strategy_memory: StrategyMemory::new(capacity),
            strategy_trained: false,
            iteration: 0,
            lr: 0.001,
            weight_decay: 1e-4,
            use_icm: false,
            fixed_players: None,
        }
    }

    fn train_from_batch(
        net: &Arc<RwLock<MLP>>,
        batch: SampleBatch,
        loss: LossType,
        lr: f32,
        weight_decay: f32,
    ) -> f32 {
        let (inputs, targets, masks, raw_weights) = batch;
        if inputs.is_empty() {
            return 0.0;
        }
        let avg = (raw_weights.iter().sum::<f32>() / raw_weights.len() as f32).max(1e-6);
        let norm: Vec<f32> = raw_weights.iter().map(|&w| w / avg).collect();
        let mut n = net.write().unwrap();
        n.train_batch(&inputs, &targets, Some(&norm), Some(&masks), loss, lr, weight_decay)
    }

    /// Trả về adv_loss trung bình. Traversal và gradient đều chạy song song.
    pub fn step_iteration<R: Rng>(
        &mut self,
        num_traversals: usize,
        train_steps_per_iter: usize,
        batch_size: usize,
        rng: &mut R,
    ) -> f32 {
        self.iteration += 1;
        let iter_weight = self.iteration as f32; // Linear CFR
        let bb_size = 100u32;
        let sb_size = 50u32;
        let use_icm = self.use_icm;
        let fixed = self.fixed_players;

        // 1. Traversal song song trên snapshot của mạng
        let snapshot: MLP = self.advantage_net.read().unwrap().clone();
        let results: Vec<(Vec<AdvantageSample>, Vec<StrategySample>)> = (0..num_traversals)
            .into_par_iter()
            .map(|_| {
                let mut trng = rand::thread_rng();
                let n = fixed.unwrap_or_else(|| trng.gen_range(2..=8));
                let button = trng.gen_range(0..n);
                let stacks = if trng.gen_bool(0.35) {
    TournamentStackSampler::sample_equal_stacks(&mut trng, n, bb_size)
} else {
    TournamentStackSampler::sample_dirichlet_stacks(&mut trng, n, bb_size)
};
                let start: Vec<f32> = stacks.iter().map(|&s| s as f32).collect();

                let mut ctx = Ctx {
                    net: &snapshot,
                    adv_out: Vec::new(),
                    strat_out: Vec::new(),
                    weight: iter_weight,
                };
                for t in 0..n {
                    let mut env = Environment::new(n, &stacks, bb_size, sb_size, button);
                    env.reset_hand(&mut trng, button);
                    let pay = Payoff::new(&start, t, use_icm, bb_size as f32);
                    traverse(&mut env, &mut ctx, &pay, &mut trng, 0);
                }
                (ctx.adv_out, ctx.strat_out)
            })
            .collect();

        for (adv, strat) in results {
            for s in adv {
                self.adv_memory.push(s, rng);
            }
            for s in strat {
                self.strategy_memory.push(s, rng);
            }
        }

        // 2. Train advantage net
        let mut adv_loss = 0.0f32;
        if self.adv_memory.len() >= batch_size && train_steps_per_iter > 0 {
            for _ in 0..train_steps_per_iter {
                let batch = self.adv_memory.sample_batch(rng, batch_size);
                adv_loss += Self::train_from_batch(
                    &self.advantage_net,
                    batch,
                    LossType::Huber,
                    self.lr,
                    self.weight_decay,
                );
            }
            adv_loss /= train_steps_per_iter as f32;
        }

        // 3. Train strategy net (average strategy)
        if train_steps_per_iter > 0 && self.strategy_memory.len() >= batch_size {
            let steps = (train_steps_per_iter / 2).max(1);
            for _ in 0..steps {
                let batch = self.strategy_memory.sample_batch(rng, batch_size);
                Self::train_from_batch(&self.strategy_net, batch, LossType::MSE, self.lr, self.weight_decay);
            }
            self.strategy_trained = true;
        }

        adv_loss
    }

    pub fn get_policy(&self, is_greedy: bool) -> DeepCFRPolicy {
        let strat = if self.strategy_trained {
            self.strategy_net.read().ok().map(|n| (*n).clone())
        } else {
            None
        };
        DeepCFRPolicy::with_strategy(Arc::clone(&self.advantage_net), strat, is_greedy)
    }

    pub fn save_models(&self, adv_path: &str, strat_path: &str) -> std::io::Result<()> {
        if let Ok(net) = self.advantage_net.read() {
            net.save_to_file(adv_path)?;
        }
        if self.strategy_trained {
            if let Ok(net) = self.strategy_net.read() {
                net.save_to_file(strat_path)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icm_sums_to_one() {
        let stacks = vec![1000.0, 500.0, 300.0, 200.0];
        let payouts = vec![0.5, 0.3, 0.2];
        let eq = icm_equity(&stacks, &payouts);
        let sum: f32 = eq.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {}", sum);
        assert!(eq[0] > eq[1] && eq[1] > eq[2]);
    }
}

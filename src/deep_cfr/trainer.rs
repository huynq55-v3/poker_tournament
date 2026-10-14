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
use crate::equity::icm::{calculate_icm_ev, get_sng_payouts};

/// Độ sâu tối đa cho traverser rẽ nhánh. 8 lượt đủ bao quát Preflop, Flop, Turn và River
/// (mỗi vòng 2 lượt hành động kể cả re-raise) mà không bị bùng nổ vì đối thủ chỉ sample 1 nhánh.
const MAX_BRANCH_DEPTH: usize = 8;

/// Hàm thưởng cho 1 traverser: Chip-EV chuẩn hóa chia cho 10 BB
struct Payoff {
    traverser: usize,
    start_icm_ev: f32,
    start_stacks: Vec<f32>,
    use_icm: bool,
    bb: f32,
}

impl Payoff {
    fn new(start: &[f32], traverser: usize, bb: f32, use_icm: bool) -> Self {
        let start_stacks = start.to_vec();
        // Heads-up (2 người) trong tournament không có áp lực ICM vì cả 2 đã cầm chắc Top 2.
        // Do đó, chỉ áp dụng ICM khi bàn có từ 3 người trở lên.
        let is_hu = start_stacks.len() <= 2;
        let start_icm_ev = if use_icm && !is_hu {
            let payouts = get_sng_payouts(start_stacks.len());
            let evs = calculate_icm_ev(&start_stacks, payouts);
            evs[traverser]
        } else {
            0.0
        };

        Payoff {
            traverser,
            start_icm_ev,
            start_stacks,
            use_icm: use_icm && !is_hu,
            bb,
        }
    }

    #[inline(always)]
    fn eval(&self, env: &Environment) -> f32 {
        // 1. Tính Chip-EV chuẩn hóa (luôn tính để làm nền tảng sát thương)
        let mut my_end_chips = env.state.players[self.traverser].stack as f32;
        if env.state.pot > 0 {
            let in_hand: Vec<usize> = env
                .state
                .players
                .iter()
                .enumerate()
                .filter(|(_, p)| p.is_in_hand())
                .map(|(i, _)| i)
                .collect();
            if in_hand.contains(&self.traverser) {
                my_end_chips += env.state.pot as f32 / in_hand.len() as f32;
            }
        }
        let my_start_chips = self.start_stacks[self.traverser];
        let chip_ev = (my_end_chips - my_start_chips) / self.bb / 10.0;

        if self.use_icm {
            let mut end_stacks: Vec<f32> = env.state.players.iter().map(|p| p.stack as f32).collect();
            if env.state.pot > 0 {
                let remaining_in_hand: Vec<usize> = env
                    .state
                    .players
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.is_in_hand())
                    .map(|(i, _)| i)
                    .collect();

                if remaining_in_hand.len() == 1 {
                    end_stacks[remaining_in_hand[0]] += env.state.pot as f32;
                } else if !remaining_in_hand.is_empty() {
                    let share = env.state.pot as f32 / remaining_in_hand.len() as f32;
                    for &idx in &remaining_in_hand {
                        end_stacks[idx] += share;
                    }
                }
            }

            let payouts = get_sng_payouts(end_stacks.len());
            let end_evs = calculate_icm_ev(&end_stacks, payouts);
            let icm_ev = (end_evs[self.traverser] - self.start_icm_ev) * 10.0;

            // Pha trộn 65% ICM + 35% Chip-EV:
            // Giữ nguyên kỷ luật sinh tồn Bubble của SNG nhưng không bị thụ động trước TAG
            0.65 * icm_ev + 0.35 * chip_ev
        } else {
            chip_ev
        }
    }
}

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
        // Nếu vượt quá độ sâu tối đa, chỉ lấy mẫu 1 hành động đi tiếp
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
        // Đối thủ chỉ lấy mẫu 1 nhánh theo chiến lược hiện tại
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
    pub fixed_players: Option<usize>,
    pub use_icm: bool,
}

impl DeepCFRTrainer {
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

    pub fn new(input_dim: usize, hidden_dims: &[usize], capacity: usize, use_icm: bool) -> Self {
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
            lr: 0.0005,
            weight_decay: 1e-4,
            fixed_players: None,
            use_icm,
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
        let norm: Vec<f32> = raw_weights.iter().map(|&w| (w / avg).clamp(0.1, 3.0)).collect();
        let mut n = net.write().unwrap();
        n.train_batch(&inputs, &targets, Some(&norm), Some(&masks), loss, lr, weight_decay)
    }

    pub fn step_iteration<R: Rng>(
        &mut self,
        num_traversals: usize,
        train_steps_per_iter: usize,
        batch_size: usize,
        rng: &mut R,
    ) -> f32 {
        self.iteration += 1;
        // Căn bậc hai (Sqrt CFR weight) giúp các iter lớn không đè bẹp hoàn toàn mẫu cũ
        let iter_weight = (self.iteration as f32).sqrt();
        let bb_size = 100u32;
        let sb_size = 50u32;
        let fixed = self.fixed_players;

        // 1. Chu kỳ Cosine Annealing (chu kỳ 50 iterations):
        // Giúp mạng liên tục thoát khỏi cực tiểu cục bộ (local minimum), duy trì tính khám phá và hội tụ sâu
        let cycle_len = 50;
        let step_in_cycle = (self.iteration - 1) % cycle_len;

        // Reset optimizer ở đầu mỗi chu kỳ để loại bỏ stale momentum tích tụ trong Adam, chống drift trọng số
        if step_in_cycle == 0 && self.iteration > 1 {
            self.advantage_net.write().unwrap().reset_optimizer();
            self.strategy_net.write().unwrap().reset_optimizer();
        }

        // Cosine Annealing Learning Rate Schedule với Warm Restarts
        let cycle_idx = (self.iteration - 1) / cycle_len;
        let cycle_max_lr = (self.lr * 0.992f32.powi(cycle_idx as i32)).max(0.00015);
        let min_lr = 0.00005;
        let cos_ratio = (std::f32::consts::PI * step_in_cycle as f32 / cycle_len as f32).cos();
        let effective_lr = min_lr + 0.5 * (cycle_max_lr - min_lr) * (1.0 + cos_ratio);

        let snapshot: MLP = self.advantage_net.read().unwrap().clone();
        let results: Vec<(Vec<AdvantageSample>, Vec<StrategySample>)> = (0..num_traversals)
            .into_par_iter()
            .map(|_| {
                let mut trng = rand::thread_rng();
                let n = fixed.unwrap_or_else(|| {
                    if trng.gen_bool(0.20) {
                        2
                    } else if trng.gen_bool(0.75) {
                        trng.gen_range(6..=8)
                    } else {
                        trng.gen_range(3..=5)
                    }
                });
                let button = trng.gen_range(0..n);
                // 50% Equal Stacks (bao gồm 8, 10, 12, 25, 50 BB) + 50% Dirichlet Stacks
                let stacks = if trng.gen_bool(0.50) {
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
                    let pay = Payoff::new(&start, t, bb_size as f32, self.use_icm);
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

        // 2. Chống Overfitting khi buffer còn nhỏ:
        // Cân bằng số steps tương đương 2 epoch trên buffer hiện có, tránh việc vài nghìn mẫu đầu bị train tới 80 epoch
        let effective_steps = if self.adv_memory.len() < 50_000 {
            (self.adv_memory.len() / batch_size * 2).clamp(10, train_steps_per_iter)
        } else {
            train_steps_per_iter
        };

        // Train Advantage Network (Huber Loss)
        let mut adv_loss = 0.0f32;
        if self.adv_memory.len() >= batch_size && effective_steps > 0 {
            for _ in 0..effective_steps {
                let batch = self.adv_memory.sample_batch(rng, batch_size);
                adv_loss += Self::train_from_batch(
                    &self.advantage_net,
                    batch,
                    LossType::Huber,
                    effective_lr,
                    self.weight_decay,
                );
            }
            adv_loss /= effective_steps as f32;
        }

        // Train Strategy Network (MSE)
        if effective_steps > 0 && self.strategy_memory.len() >= batch_size {
            let steps = (effective_steps * 2 / 3).max(1);
            for _ in 0..steps {
                let batch = self.strategy_memory.sample_batch(rng, batch_size);
                Self::train_from_batch(&self.strategy_net, batch, LossType::MSE, effective_lr, self.weight_decay);
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

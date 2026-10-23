use crate::benchmark::{
    BountyHunter, IcmSniper, LagShark, NashPushBot, NitTrap, StickyFish, ThreeBetAggro,
    TightAggressive,
};
use crate::deep_cfr::advantage_memory::{AdvantageMemory, AdvantageSample, SampleBatch};
use crate::deep_cfr::policy::DeepCFRPolicy;
use crate::deep_cfr::strategy_memory::{StrategyMemory, StrategySample};
use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::equity::icm::{calculate_icm_ev, get_sng_payouts};
use crate::neural_network::{LossType, MLP};
use crate::sampler::TournamentStackSampler;
use crate::self_play::Policy;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Độ sâu 6 vừa đủ bao phủ cả 4 vòng (Preflop -> Flop -> Turn -> River) mà chạy siêu tốc
const MAX_BRANCH_DEPTH: usize = 6;

/// Hàm thưởng kết hợp (Hybrid Tournament ICM & Chip-EV)
struct Payoff {
    traverser: usize,
    start_icm_ev: f32,
    start_stacks: Vec<f32>,
    use_icm: bool,
    bb: f32,
    is_cash_game: bool,
}

impl Payoff {
    fn new(start: &[f32], traverser: usize, bb: f32, use_icm: bool, is_cash_game: bool) -> Self {
        let start_stacks = start.to_vec();
        let use_icm_effective = use_icm && start_stacks.len() >= 2;
        let start_icm_ev = if use_icm_effective {
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
            use_icm: use_icm_effective,
            bb,
            is_cash_game,
        }
    }

    #[inline(always)]
    fn eval(&self, env: &Environment) -> f32 {
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
                let mut share = env.state.pot as f32 / in_hand.len() as f32;
                // Trừ Rake nếu là Cash Game và ván bài đi tới Flop/Turn/River
                // (No Flop No Drop: Ăn pot preflop không mất phế)
                if self.is_cash_game && env.state.street != Street::Preflop {
                    let rake = (share * 0.055).min(3.0 * self.bb);
                    share -= rake;
                }
                my_end_chips += share;
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

            // Cân bằng 50% ICM + 50% Chip-EV
            // 0.50 * icm_ev + 0.50 * chip_ev
            icm_ev
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
    sparring_tag: bool, // 🟢 Cờ kích hoạt Quân xanh TAG trên bàn đấu tập
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

/// External-sampling MCCFR kết hợp Sparring TAG & Smart Postflop Exploration
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

    if curr == pay.traverser {
        let raw = ctx.net.forward(&features);
        let strat = DeepCFRPolicy::compute_regret_matching_probs(&raw, &mask);

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
        // 🟢 TRƯỜNG HỢP 1: Bàn có Quân Xanh: Toàn bộ đối thủ (curr != traverser)
        // đều sử dụng 1 trong 8 trường phái chuẩn mực (TAG, LagShark, NashPushBot, IcmSniper, NitTrap, StickyFish, ThreeBetAggro, BountyHunter)
        // cố định theo số ghế (curr % 8) để bắt chết các hành vi Shove láo, không bỏ sót ghế nào!
        if ctx.sparring_tag {
            let bot_kind = curr % 8;
            let a = match bot_kind {
                0 => TightAggressive.select_action(&features, &mask),
                1 => LagShark.select_action(&features, &mask),
                2 => NashPushBot.select_action(&features, &mask),
                3 => IcmSniper.select_action(&features, &mask),
                4 => NitTrap.select_action(&features, &mask),
                5 => StickyFish.select_action(&features, &mask),
                6 => ThreeBetAggro.select_action(&features, &mask),
                _ => BountyHunter.select_action(&features, &mask),
            };
            return step_and_continue(env, a, ctx, pay, rng, depth);
        }

        // 🟢 TRƯỜNG HỢP 2: Đối thủ Self-Play chuẩn theo mạng hiện tại
        let raw = ctx.net.forward(&features);
        let strat = DeepCFRPolicy::compute_regret_matching_probs(&raw, &mask);

        ctx.strat_out.push(StrategySample {
            features,
            action_probs: strat.to_vec(),
            mask: mask.to_bool_array(),
            iteration_weight: ctx.weight,
        });

        // 8% Smart Exploration: Thử nghiệm các nhánh BetSmall / BetPot33 / BetPot75 để mở khóa cây Postflop,
        // tuyệt đối KHÔNG chọn bừa All-In hay Fold làm nhiễu TrashShove!
        let a = if rng.gen_bool(0.08) {
            if mask.is_valid(Action::BetSmall) && rng.gen_bool(0.35) {
                Action::BetSmall
            } else if mask.is_valid(Action::BetPot33) && rng.gen_bool(0.55) {
                Action::BetPot33
            } else if mask.is_valid(Action::BetPot75) {
                Action::BetPot75
            } else {
                sample_action(&strat, &mask, rng)
            }
        } else {
            sample_action(&strat, &mask, rng)
        };
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
    pub max_players: usize,
    pub use_icm: bool,
    pub stack_bb: u32,
    pub is_cash_game: bool,
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
            max_players: 8,
            use_icm,
            stack_bb: 100,
            is_cash_game: false,
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
        let norm: Vec<f32> = raw_weights.iter().map(|&w| (w / avg).clamp(0.05, 5.0)).collect();
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

        // Dọn dẹp các mẫu nhiễu thời kỳ đầu (Iter 1..11) đúng 1 lần tại Iter 25
        if self.iteration == 25 {
            self.adv_memory.buffer.retain(|s| s.iteration_weight >= 12.0);
            self.adv_memory.seen = self.adv_memory.buffer.len();
            self.strategy_memory.buffer.retain(|s| s.iteration_weight >= 12.0);
            self.strategy_memory.seen = self.strategy_memory.buffer.len();
        }

        // Linear CFR Weight chuẩn (Brown et al.)
        let iter_weight = self.iteration as f32;
        let bb_size = 100u32;
        let sb_size = 50u32;
        let fixed = self.fixed_players;
        let max_p = self.max_players;
        let stack_depth = self.stack_bb;
        let is_cash = self.is_cash_game;
        let use_icm = self.use_icm;

        // Lịch trình giảm Learning Rate mượt mà (Smooth Inverse Decay)
        let effective_lr = (self.lr / (1.0 + 0.004 * self.iteration as f32)).max(0.00008);

        let snapshot: MLP = self.advantage_net.read().unwrap().clone();
        let results: Vec<(Vec<AdvantageSample>, Vec<StrategySample>)> = if !is_cash && stack_depth > 0 {
            // 🟢 TOURNAMENT HYBRID MODE:
            // 1. Full Tournament Episodes (50% thông lượng): 8 người, 50 BB ban đầu, đánh đến khi hết giải.
            // 2. Strategic Multi-Depth Booster (50% thông lượng): Bơm sâu dữ liệu 50 BB (50%), 25 BB (30%), 12 BB (20%).
            let initial_bb = stack_depth;
            let num_tournaments = if num_traversals < 50 { 1 } else { (num_traversals / 50).clamp(2, 6) };
            let num_booster_hands = (num_traversals / 3).max(10);

            let tour_results: Vec<(Vec<AdvantageSample>, Vec<StrategySample>)> = (0..num_tournaments)
                .into_par_iter()
                .map(|_| {
                    let mut trng = rand::thread_rng();
                    let n = fixed.unwrap_or(max_p).clamp(2, 8);
                    let stacks = vec![initial_bb * bb_size; n];
                    let mut button = trng.gen_range(0..n);

                    let mut env = Environment::new(n, &stacks, bb_size, sb_size, button);
                    env.state.is_cash_game = false;

                    let sparring_enabled = trng.gen_bool(0.60); // 60% bàn Tournament có Quân Xanh rèn giũa toàn diện
                    let mut ctx = Ctx {
                        net: &snapshot,
                        adv_out: Vec::new(),
                        strat_out: Vec::new(),
                        weight: iter_weight,
                        sparring_tag: sparring_enabled,
                    };

                    let mut hand_idx = 0;
                    while hand_idx < 250 {
                        let alive: Vec<usize> = (0..n).filter(|&i| env.state.players[i].stack > 0).collect();
                        if alive.len() <= 1 {
                            break;
                        }

                        // Tăng Blind Level định kỳ theo tiến trình giải đấu
                        let blind_level = hand_idx / n;
                        let (sb, bb) = crate::benchmark::arena::get_tournament_blinds(blind_level);
                        env.state.sb_size = sb;
                        env.state.bb_size = bb;

                        // Chuyển Button tới người còn sống tiếp theo
                        let mut next_btn = (button + 1) % n;
                        while env.state.players[next_btn].stack == 0 {
                            next_btn = (next_btn + 1) % n;
                        }
                        button = next_btn;

                        // Ghi nhận stacks trước khi post blind để tính ICM EV / Chip EV chính xác
                        let start: Vec<f32> = env.state.players.iter().map(|p| p.stack as f32).collect();

                        env.reset_hand(&mut trng, button);

                        if env.state.street != Street::Showdown {
                            // MCCFR Chuẩn: Chọn ngẫu nhiên 1 người chơi còn sống làm Traverser duyệt cây ván này
                            let traverser = alive[trng.gen_range(0..alive.len())];
                            let pay = Payoff::new(&start, traverser, env.state.bb_size as f32, use_icm, false);
                            let mut branch = env.clone();
                            traverse(&mut branch, &mut ctx, &pay, &mut trng, 0);

                            // Chơi ván bài thực tế để cập nhật stack và loại người chơi bị hết chip
                            let mut guard = 0;
                            while env.state.street != Street::Showdown && guard < 400 {
                                guard += 1;
                                let cur = env.state.current_player_idx;
                                let mask = env.get_action_mask();
                                if mask.0 == 0 {
                                    break;
                                }
                                let feats = env.state.encode_features(cur);
                                let action = if sparring_enabled && (cur != traverser) {
                                    let bot_kind = cur % 8;
                                    match bot_kind {
                                        0 => TightAggressive.select_action(&feats, &mask),
                                        1 => LagShark.select_action(&feats, &mask),
                                        2 => NashPushBot.select_action(&feats, &mask),
                                        3 => IcmSniper.select_action(&feats, &mask),
                                        4 => NitTrap.select_action(&feats, &mask),
                                        5 => StickyFish.select_action(&feats, &mask),
                                        6 => ThreeBetAggro.select_action(&feats, &mask),
                                        _ => BountyHunter.select_action(&feats, &mask),
                                    }
                                } else {
                                    let raw = snapshot.forward(&feats);
                                    let probs = DeepCFRPolicy::compute_regret_matching_probs(&raw, &mask);
                                    sample_action(&probs, &mask, &mut trng)
                                };
                                match env.step(action) {
                                    Ok(true) | Err(_) => break,
                                    Ok(false) => {}
                                }
                            }
                        }

                        hand_idx += 1;
                    }

                    (ctx.adv_out, ctx.strat_out)
                })
                .collect();

            let booster_results: Vec<(Vec<AdvantageSample>, Vec<StrategySample>)> = (0..num_booster_hands)
                .into_par_iter()
                .map(|_| {
                    let mut trng = rand::thread_rng();
                    // 🟢 DÀNH RIÊNG 35% BOOSTER CHO HEADS-UP (1-vs-1) ĐỂ HUẤN LUYỆN SÁT THỦ HEADS-UP:
                    let is_hu = trng.gen_bool(0.35);
                    let (n, target_depth) = if is_hu {
                        // Heads-Up: Quét đều mọi tầng stack từ Deep Stack đến Push/Fold
                        let hu_depth = match trng.gen_range(0..5) {
                            0 => 50u32,
                            1 => 35u32,
                            2 => 20u32,
                            3 => 12u32,
                            _ => 8u32,
                        };
                        (2, hu_depth)
                    } else {
                        // 65% còn lại cho các giai đoạn Multiway trong giải đấu:
                        // Phân bổ có chủ đích: 50 BB (40%), 25 BB (30%), 12 BB (15%), 8 BB (15%)
                        let roll: f32 = trng.gen();
                        let depth = if roll < 0.40 {
                            50u32
                        } else if roll < 0.70 {
                            25u32
                        } else if roll < 0.85 {
                            12u32
                        } else {
                            8u32
                        };
                        let players = match depth {
                            50 => max_p.clamp(6, 8),
                            25 => trng.gen_range(5..=max_p.clamp(6, 8)),
                            12 => trng.gen_range(3..=6.min(max_p)),
                            _ => trng.gen_range(3..=max_p),
                        };
                        (players, depth)
                    };

                    let button = trng.gen_range(0..n);
                    let stacks = if is_hu {
                        // Trong Heads-Up:
                        // - 50% bàn Chipleader đè Short-stack (Dirichlet Asymmetric, ví dụ 70% vs 30% chip)
                        //   để mạng học cách dùng ưu thế stack lớn cướp blind và trừng phạt đối thủ!
                        // - 50% bàn Equal Stack theo target_depth
                        if trng.gen_bool(0.50) {
                            let total = target_depth * 2 * bb_size;
                            let leader_share: f32 = trng.gen_range(0.55..0.85);
                            let s0 = ((total as f32) * leader_share).round() as u32;
                            let s1 = total.saturating_sub(s0).max(4 * bb_size);
                            if trng.gen_bool(0.50) {
                                vec![s0, s1]
                            } else {
                                vec![s1, s0]
                            }
                        } else {
                            vec![target_depth * bb_size; 2]
                        }
                    } else if trng.gen_bool(0.40) {
                        vec![target_depth * bb_size; n]
                    } else {
                        TournamentStackSampler::sample_tournament_progression_stacks(
                            &mut trng,
                            n,
                            target_depth,
                            max_p,
                            bb_size,
                        )
                    };
                    let start: Vec<f32> = stacks.iter().map(|&s| s as f32).collect();

                    let mut ctx = Ctx {
                        net: &snapshot,
                        adv_out: Vec::new(),
                        strat_out: Vec::new(),
                        weight: iter_weight,
                        sparring_tag: trng.gen_bool(0.70), // 70% Booster Hands có Quân Xanh trừng phạt toàn diện
                    };

                    for t in 0..n {
                        let mut env = Environment::new(n, &stacks, bb_size, sb_size, button);
                        env.state.is_cash_game = false;
                        env.reset_hand(&mut trng, button);
                        let pay = Payoff::new(&start, t, bb_size as f32, use_icm, false);
                        let mut branch = env.clone();
                        traverse(&mut branch, &mut ctx, &pay, &mut trng, 0);
                    }
                    (ctx.adv_out, ctx.strat_out)
                })
                .collect();

            tour_results.into_iter().chain(booster_results.into_iter()).collect()
        } else {
            // Cash Game mode hoặc Random Dirichlet Stack mode (Final Table)
            (0..num_traversals)
                .into_par_iter()
                .map(|_| {
                    let mut trng = rand::thread_rng();
                    let n = fixed.unwrap_or_else(|| {
                        if max_p <= 4 {
                            if trng.gen_bool(0.35) {
                                2
                            } else if trng.gen_bool(0.25) {
                                3
                            } else {
                                max_p
                            }
                        } else if max_p <= 6 {
                            if trng.gen_bool(0.35) {
                                2
                            } else if trng.gen_bool(0.45) {
                                max_p
                            } else {
                                trng.gen_range(3..=max_p.saturating_sub(1).max(3))
                            }
                        } else {
                            if trng.gen_bool(0.35) {
                                2
                            } else if trng.gen_bool(0.45) {
                                trng.gen_range(7..=8)
                            } else {
                                trng.gen_range(3..=6)
                            }
                        }
                    });
                    let button = trng.gen_range(0..n);
                    let stacks = if is_cash {
                        TournamentStackSampler::sample_cash_game_stacks(&mut trng, n, bb_size)
                    } else if trng.gen_bool(0.25) {
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
                        sparring_tag: trng.gen_bool(0.50),
                    };
                    for t in 0..n {
                        let mut env = Environment::new(n, &stacks, bb_size, sb_size, button);
                        if is_cash {
                            env.state.is_cash_game = true;
                        }
                        env.reset_hand(&mut trng, button);
                        let pay = Payoff::new(&start, t, bb_size as f32, use_icm, is_cash);
                        traverse(&mut env, &mut ctx, &pay, &mut trng, 0);
                    }
                    (ctx.adv_out, ctx.strat_out)
                })
                .collect()
        };

        for (adv, strat) in results {
            for s in adv {
                self.adv_memory.push(s, rng);
            }
            for s in strat {
                self.strategy_memory.push(s, rng);
            }
        }

        let effective_steps = if train_steps_per_iter == 0 {
            0
        } else if self.adv_memory.len() < 50_000 {
            (self.adv_memory.len() / batch_size * 2).clamp(15.min(train_steps_per_iter), train_steps_per_iter)
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

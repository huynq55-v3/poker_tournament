//! Bước 3 của bot RL: tự chơi hàng loạt ván heads-up song song, gom quyết định thành batch.
//!
//! Mỗi vòng lặp: mọi bàn đang chờ người chơi do mô hình điều khiển được gom thành **một batch**
//! cho mô hình chạy một lần (GPU hiệu quả hơn nhiều so với chạy từng trạng thái). Bàn nào xong ván
//! thì ghi lại quỹ đạo và chia ván mới cho tới khi đủ số ván.
//!
//! Luật cố định: heads-up, mỗi bên 100 BB đầu mỗi ván, button luân phiên.
//! Phần thưởng: chip thắng/thua cuối ván, đơn vị 100 BB (cùng đơn vị với đầu value).

use crate::environment::{Action, ActionMask, Environment, NUM_ACTIONS};
use crate::rl::encoding::{encode, EncodedState};
use crate::rl::model::PokerTransformer;
use burn::tensor::Device;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Big blind (chip). Stack khởi điểm = 100 BB.
pub const BB: u32 = 100;
pub const START_STACK: u32 = 100 * BB;
/// 1 đơn vị phần thưởng = 100 BB.
pub const REWARD_UNIT: f32 = START_STACK as f32;

/// Một quyết định của người chơi được ghi lại để train.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: EncodedState,
    pub mask: ActionMask,
    pub action: u8,
    /// log π_old(a|s) lúc tự chơi (PPO cần để tính tỉ lệ π_new / π_old).
    pub log_prob: f32,
    /// V_old(s) lúc tự chơi.
    pub value: f32,
}

/// Toàn bộ quyết định của một người chơi trong một ván, theo thứ tự thời gian.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub steps: Vec<Transition>,
    /// Chip thắng/thua cuối ván của người chơi này, đơn vị 100 BB.
    pub reward: f32,
}

/// Người chơi ngồi một ghế.
#[derive(Clone, Copy)]
pub enum Agent<'a> {
    /// Chọn hành động theo phân phối policy của mô hình.
    Model(&'a PokerTransformer),
    /// Chọn đều ngẫu nhiên trong các hành động hợp lệ.
    Random,
    /// Luôn check/call.
    AlwaysCall,
    /// Bot đơn giản dựa vào sức mạnh bài (Chen preflop, loại tay đã thành postflop). Làm mốc đo tiến bộ.
    Heuristic,
}

/// Đối thủ của learner khi thu thập dữ liệu.
#[derive(Clone, Copy)]
pub enum Opponent<'a> {
    /// Learner tự chơi với chính nó; quyết định của cả hai ghế đều được ghi lại.
    SelfPlay,
    /// Learner đấu với một agent khác (vd. bản cũ trong kho); chỉ ghi quyết định của learner.
    Fixed(Agent<'a>),
}

#[derive(Debug, Clone)]
pub struct RolloutConfig {
    /// Tổng số ván cần chơi.
    pub hands: usize,
    /// Số bàn chạy song song (= kích thước batch tối đa khi suy luận).
    pub parallel: usize,
    pub seed: u64,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        RolloutConfig { hands: 1024, parallel: 256, seed: 0 }
    }
}

/// Thống kê một lượt thu thập.
#[derive(Debug, Clone, Default)]
pub struct RolloutStats {
    pub hands: usize,
    pub decisions: usize,
    /// Tần suất từng hành động của learner.
    pub action_counts: [usize; NUM_ACTIONS],
    /// Tổng phần thưởng của learner (ghế learner khi đấu Fixed; ghế 0 khi SelfPlay), đơn vị 100 BB.
    pub learner_reward_sum: f64,
    pub learner_reward_sq_sum: f64,
    pub seconds: f64,
}

impl RolloutStats {
    /// Lãi trung bình của learner theo BB/100 ván.
    pub fn learner_bb_per_100(&self) -> f64 {
        self.learner_reward_sum / self.hands.max(1) as f64 * 100.0 * 100.0
    }
}

#[derive(Debug, Clone, Default)]
pub struct RolloutBatch {
    pub trajectories: Vec<Trajectory>,
    pub stats: RolloutStats,
}

/// Một bàn đang chạy.
struct Table {
    env: Environment,
    /// Ghế nào là learner (khi đấu Fixed). Khi SelfPlay cả hai ghế đều là learner.
    learner_seat: usize,
    steps: [Vec<Transition>; 2],
}

fn new_hand(rng: &mut StdRng, hand_idx: usize, learner_seat: usize) -> Table {
    let mut env = Environment::new(2, &[START_STACK, START_STACK], BB, BB / 2, 0);
    env.state.is_cash_game = true;
    env.reset_hand(rng, hand_idx % 2);
    Table { env, learner_seat, steps: [Vec::new(), Vec::new()] }
}

/// Lấy mẫu một hành động hợp lệ từ phân phối `probs` (đã chuẩn hóa lại trên các hành động hợp lệ).
pub fn sample_action(probs: &[f32; NUM_ACTIONS], mask: ActionMask, rng: &mut impl Rng) -> usize {
    let total: f32 = (0..NUM_ACTIONS).filter(|&a| mask.is_index_valid(a)).map(|a| probs[a].max(0.0)).sum();
    if total <= 0.0 || !total.is_finite() {
        let valid = mask.valid_actions();
        return valid[rng.gen_range(0..valid.len())].to_index();
    }
    let mut r = rng.gen::<f32>() * total;
    let mut last = None;
    for a in 0..NUM_ACTIONS {
        if !mask.is_index_valid(a) {
            continue;
        }
        last = Some(a);
        r -= probs[a].max(0.0);
        if r <= 0.0 {
            return a;
        }
    }
    last.expect("mask rỗng")
}

/// Một quyết định của agent: hành động, log π(a|s), V(s), và trạng thái đã mã hóa (chỉ với mô hình).
struct Decision {
    action: usize,
    log_prob: f32,
    value: f32,
    state: Option<EncodedState>,
}

/// Hành động của bot `Heuristic` cho người đang đến lượt.
pub fn heuristic_action(env: &Environment) -> usize {
    use crate::environment::state::chen_score;
    use crate::environment::Street;
    use crate::poker_core::{evaluate_7_cards, HandCategory};

    let st = &env.state;
    let p = st.current_player_idx;
    let mask = env.get_action_mask();
    let me = &st.players[p];
    let facing_bet = st.current_highest_bet > me.current_bet;
    let [Some(c0), Some(c1)] = me.hole_cards else {
        return Action::CheckCall.to_index();
    };
    // 2 = mạnh (raise), 1 = vừa (call), 0 = yếu (check/fold)
    let strength = if st.street == Street::Preflop {
        let chen = chen_score(c0, c1);
        if chen >= 10.0 { 2 } else if chen >= 6.0 { 1 } else { 0 }
    } else {
        let mut cards = vec![c0, c1];
        cards.extend_from_slice(&st.community_cards);
        match evaluate_7_cards(&cards).category() {
            HandCategory::HighCard => 0,
            HandCategory::OnePair => 1,
            _ => 2,
        }
    };
    match strength {
        2 if mask.is_valid(Action::BetPot75) => Action::BetPot75.to_index(),
        2 | 1 => Action::CheckCall.to_index(),
        _ if facing_bet && mask.is_valid(Action::Fold) => Action::Fold.to_index(),
        _ => Action::CheckCall.to_index(),
    }
}

/// Cho `agent` chọn hành động cho người đang đến lượt ở mỗi bàn trong `envs` (một batch).
fn agent_act(agent: Agent, envs: &[&Environment], device: &Device, rng: &mut StdRng) -> Vec<Decision> {
    if envs.is_empty() {
        return Vec::new();
    }
    let masks: Vec<ActionMask> = envs.iter().map(|e| e.get_action_mask()).collect();
    match agent {
        Agent::Model(m) => {
            let enc: Vec<EncodedState> = envs.iter().map(|e| encode(&e.state, e.state.current_player_idx)).collect();
            let refs: Vec<&EncodedState> = enc.iter().collect();
            let out = m.infer(&refs, &masks, device);
            out.iter()
                .zip(&masks)
                .zip(enc)
                .map(|((inf, &mask), state)| {
                    let a = sample_action(&inf.probs, mask, rng);
                    Decision { action: a, log_prob: inf.probs[a].max(1e-12).ln(), value: inf.value, state: Some(state) }
                })
                .collect()
        }
        Agent::Random => masks
            .iter()
            .map(|m| {
                let valid = m.valid_actions();
                let a = valid[rng.gen_range(0..valid.len())].to_index();
                Decision { action: a, log_prob: -(valid.len() as f32).ln(), value: 0.0, state: None }
            })
            .collect(),
        Agent::AlwaysCall => masks
            .iter()
            .map(|_| Decision { action: Action::CheckCall.to_index(), log_prob: 0.0, value: 0.0, state: None })
            .collect(),
        Agent::Heuristic => envs
            .iter()
            .map(|e| Decision { action: heuristic_action(e), log_prob: 0.0, value: 0.0, state: None })
            .collect(),
    }
}

/// Chơi `cfg.hands` ván giữa learner và `opponent`, trả về các quỹ đạo của learner.
pub fn collect(learner: &PokerTransformer, opponent: Opponent, cfg: &RolloutConfig, device: &Device) -> RolloutBatch {
    let t0 = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let self_play = matches!(opponent, Opponent::SelfPlay);
    let parallel = cfg.parallel.max(1).min(cfg.hands.max(1));

    let mut started = 0usize;
    let mut tables: Vec<Table> = Vec::with_capacity(parallel);
    while tables.len() < parallel && started < cfg.hands {
        tables.push(new_hand(&mut rng, started, started % 2));
        started += 1;
    }
    let mut out = RolloutBatch::default();

    while !tables.is_empty() {
        // Phân bàn theo người phải hành động.
        let mut learner_idx = Vec::new();
        let mut opp_idx = Vec::new();
        for (i, t) in tables.iter().enumerate() {
            let p = t.env.state.current_player_idx;
            if self_play || p == t.learner_seat {
                learner_idx.push(i);
            } else {
                opp_idx.push(i);
            }
        }

        let mut decisions: Vec<(usize, Decision)> = Vec::new();
        let envs: Vec<&Environment> = learner_idx.iter().map(|&i| &tables[i].env).collect();
        decisions.extend(learner_idx.iter().copied().zip(agent_act(Agent::Model(learner), &envs, device, &mut rng)));
        if let Opponent::Fixed(agent) = opponent {
            let envs: Vec<&Environment> = opp_idx.iter().map(|&i| &tables[i].env).collect();
            let acts = agent_act(agent, &envs, device, &mut rng);
            // Không ghi quyết định của đối thủ.
            decisions.extend(opp_idx.iter().copied().zip(acts.into_iter().map(|d| Decision { state: None, ..d })));
        }

        // Thực hiện hành động; bàn nào xong ván thì ghi quỹ đạo và chia ván mới (hoặc bỏ bàn).
        let mut finished = Vec::new();
        for (i, Decision { action: a, log_prob, value, state: enc }) in decisions {
            let t = &mut tables[i];
            let p = t.env.state.current_player_idx;
            let mask = t.env.get_action_mask();
            if let Some(state) = enc {
                out.stats.decisions += 1;
                out.stats.action_counts[a] += 1;
                t.steps[p].push(Transition { state, mask, action: a as u8, log_prob, value });
            }
            let action = Action::from_index(a).expect("action index");
            let done = t.env.step(action).expect("mask chỉ cho hành động hợp lệ");
            if done {
                finished.push(i);
            }
        }

        finished.sort_unstable();
        for &i in finished.iter().rev() {
            let t = &mut tables[i];
            let rewards: [f32; 2] =
                std::array::from_fn(|s| (t.env.state.players[s].stack as f32 - START_STACK as f32) / REWARD_UNIT);
            assert!((rewards[0] + rewards[1]).abs() < 1e-6, "heads-up phải zero-sum");
            let stat_seat = if self_play { 0 } else { t.learner_seat };
            out.stats.hands += 1;
            out.stats.learner_reward_sum += rewards[stat_seat] as f64;
            out.stats.learner_reward_sq_sum += (rewards[stat_seat] as f64).powi(2);
            for s in 0..2 {
                let steps = std::mem::take(&mut t.steps[s]);
                if !steps.is_empty() {
                    out.trajectories.push(Trajectory { steps, reward: rewards[s] });
                }
            }
            if started < cfg.hands {
                let seat = started % 2;
                tables[i] = new_hand(&mut rng, started, seat);
                started += 1;
            } else {
                tables.swap_remove(i);
            }
        }
    }

    out.stats.seconds = t0.elapsed().as_secs_f64();
    out
}

/// Kết quả đấu đánh giá.
#[derive(Debug, Clone, Copy)]
pub struct EvalResult {
    /// Số cặp ván (mỗi cặp = cùng bộ bài, hai bên đổi ghế).
    pub pairs: usize,
    /// Lãi trung bình của `a` theo BB/100 ván.
    pub bb_per_100: f64,
    /// Sai số chuẩn của `bb_per_100`.
    pub se_bb_per_100: f64,
}

impl EvalResult {
    /// Khoảng tin cậy 95%.
    pub fn ci95(&self) -> (f64, f64) {
        (self.bb_per_100 - 1.96 * self.se_bb_per_100, self.bb_per_100 + 1.96 * self.se_bb_per_100)
    }
}

/// Đấu lặp (duplicate) `a` với `b`: mỗi bộ bài được chơi hai lần, hai bên đổi ghế, để triệt tiêu
/// phần lớn may rủi do lá bài. Trả về lãi của `a`.
pub fn evaluate(a: Agent, b: Agent, pairs: usize, parallel: usize, seed: u64, device: &Device) -> EvalResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let deck_seeds: Vec<u64> = (0..pairs).map(|_| rng.gen()).collect();
    // Mỗi phần tử: (bàn, chỉ số cặp, ghế của a)
    let mut pair_sum = vec![0f64; pairs];
    let mut queue: Vec<(usize, usize)> = (0..pairs).flat_map(|p| [(p, 0usize), (p, 1usize)]).collect();
    queue.reverse();
    let mut live: Vec<(Environment, usize, usize)> = Vec::new();
    let spawn = |p: usize, seat_a: usize| {
        let mut env = Environment::new(2, &[START_STACK, START_STACK], BB, BB / 2, 0);
        env.state.is_cash_game = true;
        // Cùng hạt giống bộ bài và cùng button cho cả hai ván của cặp → ghế đổi, bài giữ nguyên.
        env.reset_hand(&mut StdRng::seed_from_u64(deck_seeds[p]), p % 2);
        (env, p, seat_a)
    };
    while live.len() < parallel.max(1) {
        match queue.pop() {
            Some((p, s)) => live.push(spawn(p, s)),
            None => break,
        }
    }

    while !live.is_empty() {
        let mut idx_a = Vec::new();
        let mut idx_b = Vec::new();
        for (i, (env, _, seat_a)) in live.iter().enumerate() {
            if env.state.current_player_idx == *seat_a {
                idx_a.push(i);
            } else {
                idx_b.push(i);
            }
        }
        let mut chosen: Vec<(usize, usize)> = Vec::new();
        for (agent, idxs) in [(a, &idx_a), (b, &idx_b)] {
            let envs: Vec<&Environment> = idxs.iter().map(|&i| &live[i].0).collect();
            let acts = agent_act(agent, &envs, device, &mut rng);
            chosen.extend(idxs.iter().copied().zip(acts.into_iter().map(|d| d.action)));
        }
        let mut finished = Vec::new();
        for (i, act) in chosen {
            if live[i].0.step(Action::from_index(act).unwrap()).expect("hành động hợp lệ") {
                finished.push(i);
            }
        }
        finished.sort_unstable();
        for &i in finished.iter().rev() {
            let (env, p, seat_a) = &live[i];
            pair_sum[*p] += (env.state.players[*seat_a].stack as f64 - START_STACK as f64) / BB as f64;
            match queue.pop() {
                Some((p, s)) => live[i] = spawn(p, s),
                None => {
                    live.swap_remove(i);
                }
            }
        }
    }

    // Mỗi cặp: lãi trung bình mỗi ván (BB) của a.
    let per_hand: Vec<f64> = pair_sum.iter().map(|s| s / 2.0).collect();
    let n = per_hand.len().max(1) as f64;
    let mean = per_hand.iter().sum::<f64>() / n;
    let var = per_hand.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    EvalResult { pairs, bb_per_100: mean * 100.0, se_bb_per_100: (var / n).sqrt() * 100.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::model::PokerTransformerConfig;

    fn tiny_model(dev: &Device) -> PokerTransformer {
        PokerTransformerConfig::new().with_d_model(16).with_d_ff(32).with_n_heads(2).with_n_layers(1).init(dev)
    }

    #[test]
    fn self_play_trajectories_are_consistent_and_zero_sum() {
        let dev = Device::flex();
        let model = tiny_model(&dev);
        let cfg = RolloutConfig { hands: 64, parallel: 16, seed: 1 };
        let batch = collect(&model, Opponent::SelfPlay, &cfg, &dev);
        assert_eq!(batch.stats.hands, 64);
        let total_steps: usize = batch.trajectories.iter().map(|t| t.steps.len()).sum();
        assert_eq!(total_steps, batch.stats.decisions);
        assert!(batch.stats.decisions >= 64, "mỗi ván có ít nhất 1 quyết định");

        // Zero-sum từng ván được assert trong `collect`. Ghế không có quyết định nào (vd. BB khi SB fold
        // ngay preflop) không tạo quỹ đạo, nên tổng phần thưởng các quỹ đạo không nhất thiết = 0.
        assert!(batch.trajectories.len() <= 2 * 64);
        for t in &batch.trajectories {
            assert!(t.reward.abs() <= 1.0 + 1e-6, "không thể thắng/thua quá 100 BB");
            for s in &t.steps {
                assert!(s.mask.is_index_valid(s.action as usize), "hành động không hợp lệ được ghi lại");
                assert!(s.log_prob <= 0.0 && s.log_prob.is_finite());
                assert!(s.value.is_finite());
            }
        }
    }

    #[test]
    fn fixed_opponent_records_only_learner_and_is_deterministic() {
        let dev = Device::flex();
        let model = tiny_model(&dev);
        let cfg = RolloutConfig { hands: 40, parallel: 8, seed: 2 };
        let a = collect(&model, Opponent::Fixed(Agent::AlwaysCall), &cfg, &dev);
        let b = collect(&model, Opponent::Fixed(Agent::AlwaysCall), &cfg, &dev);
        assert!(a.trajectories.len() <= 40, "chỉ một ghế (learner) được ghi mỗi ván");
        assert_eq!(a.stats.decisions, b.stats.decisions, "cùng seed phải cho cùng kết quả");
        assert_eq!(a.stats.learner_reward_sum, b.stats.learner_reward_sum);
        // AlwaysCall không bao giờ fold → chỉ learner có hành động được ghi (đủ cho mọi ván có lượt learner).
        assert!(a.stats.decisions > 0);
    }

    #[test]
    fn duplicate_eval_of_identical_deterministic_agents_is_exactly_zero() {
        let dev = Device::flex();
        let r = evaluate(Agent::AlwaysCall, Agent::AlwaysCall, 200, 32, 3, &dev);
        assert_eq!(r.bb_per_100, 0.0, "cùng bài, đổi ghế, cùng chiến lược → lãi đúng bằng 0");
        assert_eq!(r.se_bb_per_100, 0.0);
    }

    #[test]
    fn heuristic_beats_bots_that_ignore_their_cards() {
        // Luôn call và ngẫu nhiên đều không nhìn bài → hòa nhau; bot nhìn bài phải thắng cả hai.
        let dev = Device::flex();
        for (name, opp) in [("Random", Agent::Random), ("AlwaysCall", Agent::AlwaysCall)] {
            let r = evaluate(Agent::Heuristic, opp, 20_000, 256, 4, &dev);
            let (lo, _) = r.ci95();
            assert!(lo > 0.0, "Heuristic vs {name}: {:.1} ± {:.1} BB/100", r.bb_per_100, r.se_bb_per_100);
            println!("Heuristic vs {name}: {:+.1} ± {:.1} BB/100", r.bb_per_100, r.se_bb_per_100);
        }
    }
}

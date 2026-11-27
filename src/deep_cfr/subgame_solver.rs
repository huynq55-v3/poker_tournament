//! Subgame Solver (Depth-Limited Solving / Lookahead Search) cho Poker AI
//!
//! Dựa trên nguyên lý Continual Resolving & Subgame Solving chuẩn GTO của DeepStack, Libratus và ReBeL:
//! 1. **Bayesian Range Weighting (0% hardcode)**: Tính trọng số range đối thủ (và public range của Hero)
//!    trực tiếp từ likelihood theo lịch sử `action_history` của chính Blueprint Neural Network (hoặc uniform nếu chưa có lịch sử).
//! 2. **2-Sided Exact Range-vs-Range Linear CFR+ (Heads-Up)** & **N-Player Multiway External-Sampling CFR+**:
//!    Cả Hero lẫn Đối thủ cùng cập nhật regret CFR+ trong cây con (không khóa cứng đối thủ 1 chiều vào Blueprint,
//!    không clairvoyance), triệt tiêu hiện tượng "lạm dụng over-fold giả lập của Blueprint" bằng cân bằng Nash cục bộ.
//! 3. **Chẩn đoán minh bạch (`SubgameDiagnostics`)**: Báo cáo rõ solver mode, IP/OOP, Bayesian ESS, Showdown Equity,
//!    sai số chuẩn `±SE BB`, và tỉ lệ Fold/Call/Raise tức thì của đối thủ trước từng kích cỡ cược của Hero.

use crate::deep_cfr::policy::DeepCFRPolicy;
use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::{GameState, Street};
use crate::equity::icm::{calculate_icm_ev, get_sng_payouts};
use crate::equity::{equity_vs_random, preflop_table};
use crate::neural_network::MLP;
use crate::poker_core::card::Card;
use crate::poker_core::deck::Deck;
use crate::poker_core::evaluator::{evaluate_7_cards, HandRank};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const BAYESIAN_PROB_FLOOR: f32 = 1e-3;
const HU_RANGE_SIZE: usize = 28;
const HU_MAX_TREE_NODES: usize = 384;
const HU_RUNOUT_SAMPLES: usize = 16;
const MULTIWAY_RANGE_SIZE: usize = 24;

/// Lý do kích hoạt tìm kiếm theo độ sâu
#[derive(Clone, Debug, PartialEq)]
pub enum SearchTrigger {
    HeadsUpDuel,
    UnusualOverbet { bet_pot_ratio: f32 },
    HighStakesPot { pot_bb: f32 },
    CrucialPostflopStreet { street: Street },
    DeepStackCommitment { to_call_ratio: f32 },
    ManualForced,
    FastPolicyFallback,
}

impl std::fmt::Display for SearchTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchTrigger::HeadsUpDuel => write!(f, "⚔️ Đối đầu 1-vs-1 Heads-Up (Chỉ còn 2 người trên bàn)"),
            SearchTrigger::UnusualOverbet { bet_pot_ratio } => {
                write!(f, "Đối thủ Overbet dị ({:.1}% pot)", bet_pot_ratio * 100.0)
            }
            SearchTrigger::HighStakesPot { pot_bb } => {
                write!(f, "Pot lớn ({:.1} BB)", pot_bb)
            }
            SearchTrigger::CrucialPostflopStreet { street } => {
                write!(f, "Vòng cược nhạy cảm ({:?})", street)
            }
            SearchTrigger::DeepStackCommitment { to_call_ratio } => {
                write!(f, "Cú cược đe dọa stack ({:.1}% stack)", to_call_ratio * 100.0)
            }
            SearchTrigger::ManualForced => write!(f, "Chỉ định tìm kiếm thủ công (--search)"),
            SearchTrigger::FastPolicyFallback => write!(f, "Chạy mạng thuần (Không cần search)"),
        }
    }
}

/// Cấu hình cho Depth-Limited Solver
#[derive(Clone, Debug)]
pub struct SubgameSolverConfig {
    pub max_depth: usize,               // Độ sâu tối đa số actions (mặc định 3)
    pub iterations: usize,              // Số vòng lặp CFR+ (mặc định 60)
    pub min_pot_bb_trigger: f32,        // Kích hoạt khi pot >= X BB (mặc định 12.0)
    pub unusual_bet_ratio_trigger: f32, // Kích hoạt khi cược > 1.10 pot (Overbet)
    pub search_turn_river: bool,        // Tự động tìm kiếm ở Turn và River
    pub use_icm: bool,                  // Đánh giá EV theo Tournament ICM (mặc định true)
}

impl Default for SubgameSolverConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            iterations: 60,
            min_pot_bb_trigger: 12.0,
            unusual_bet_ratio_trigger: 1.10,
            search_turn_river: true,
            use_icm: true,
        }
    }
}

impl SubgameSolverConfig {
    /// Cấu hình tối ưu cho Lounge Cash Game (Chip-EV, Deep Stack 12-300+ BB, 2-10 Max)
    pub fn lounge_cash_default() -> Self {
        Self {
            max_depth: 4,
            iterations: 120,
            min_pot_bb_trigger: 15.0,
            unusual_bet_ratio_trigger: 1.10,
            search_turn_river: true,
            use_icm: false,
        }
    }
}

/// Thông tin chẩn đoán chi tiết từ Subgame Solver để debug Live & kiểm định GTO
#[derive(Clone, Debug, Default)]
pub struct SubgameDiagnostics {
    pub solver_kind: &'static str,
    pub num_in_hand: usize,
    pub hero_is_ip: bool,
    pub opp_range_size: usize,
    pub opp_bayesian_ess: f32,
    pub bayesian_history_steps: usize,
    pub hero_showdown_eq: f32,
    pub action_ev_se: [f32; NUM_ACTIONS],
    pub opp_fold_pct: [f32; NUM_ACTIONS],
    pub opp_call_pct: [f32; NUM_ACTIONS],
    pub opp_raise_pct: [f32; NUM_ACTIONS],
}

/// Kết quả tính toán từ Subgame Solver
#[derive(Clone, Debug)]
pub struct SubgameResult {
    pub best_action: Action,
    pub action_probs: [f32; NUM_ACTIONS],
    pub action_evs: [f32; NUM_ACTIONS],
    pub trigger: SearchTrigger,
    pub iterations_run: usize,
    pub duration_ms: f64,
    pub diagnostics: SubgameDiagnostics,
}

#[inline]
fn blueprint_probs_for_state(
    state: &GameState,
    seat: usize,
    mask: &ActionMask,
    adv_net: &MLP,
    strat_net: Option<&MLP>,
) -> [f32; NUM_ACTIONS] {
    let features = state.encode_features(seat);
    if let Some(strat_mlp) = strat_net {
        let raw = strat_mlp.forward(&features);
        DeepCFRPolicy::compute_strategy_probs(&raw, mask)
    } else {
        let raw = adv_net.forward(&features);
        DeepCFRPolicy::compute_regret_matching_probs(&raw, mask)
    }
}

/// Tái dựng các snapshot `(GameState, ActionMask, Action)` trước từng hành động của `target_seat`
/// từ `env.state.action_history` (thuần túy từ trạng thái bàn chơi, không dùng rule cứng).
fn extract_player_history_snapshots(
    env: &Environment,
    target_seat: usize,
) -> Vec<(GameState, ActionMask, Action)> {
    let history = &env.state.action_history;
    if history.is_empty() {
        return Vec::new();
    }

    let n = env.state.num_players;
    let mut hist_spent = vec![0u32; n];
    for rec in history {
        if rec.player_idx < n {
            hist_spent[rec.player_idx] = hist_spent[rec.player_idx].saturating_add(rec.amount);
        }
    }

    let mut replay = env.state.clone();
    let mut sum_prior_contribs = 0u32;
    for (i, p) in replay.players.iter_mut().enumerate() {
        let cur_total = p.total_contributed.max(p.current_bet);
        let prior_contrib = cur_total.saturating_sub(hist_spent[i]);
        sum_prior_contribs = sum_prior_contribs.saturating_add(prior_contrib);

        p.stack = p.stack.saturating_add(hist_spent[i]);
        p.current_bet = 0;
        p.total_contributed = prior_contrib;
        p.is_all_in = false;
        if history.iter().any(|r| r.player_idx == i) {
            p.is_folded = false;
        }
    }

    let total_hist_spent: u32 = hist_spent.iter().sum();
    replay.pot = env.state.pot.saturating_sub(total_hist_spent).max(sum_prior_contribs);
    replay.current_highest_bet = 0;
    replay.min_raise = replay.bb_size.max(1);
    replay.action_history.clear();

    let first_street = history.first().map(|r| r.street).unwrap_or(Street::Preflop);
    replay.street = first_street;
    let keep_cards = match first_street {
        Street::Preflop => 0,
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River | Street::Showdown => 5,
    };
    replay.community_cards = env
        .state
        .community_cards
        .iter()
        .take(keep_cards)
        .copied()
        .collect();

    let mut snapshots = Vec::new();
    for rec in history {
        if rec.player_idx >= n {
            continue;
        }
        if rec.street != replay.street {
            replay.street = rec.street;
            replay.current_highest_bet = 0;
            replay.min_raise = replay.bb_size.max(1);
            for p in &mut replay.players {
                p.current_bet = 0;
            }
            let cnt = match rec.street {
                Street::Preflop => 0,
                Street::Flop => 3,
                Street::Turn => 4,
                Street::River | Street::Showdown => 5,
            };
            replay.community_cards = env
                .state
                .community_cards
                .iter()
                .take(cnt)
                .copied()
                .collect();
        }

        if rec.player_idx == target_seat && rec.action != Action::Fold {
            let mut tmp_env = Environment {
                state: replay.clone(),
                deck: Deck::new(),
                street_has_acted: vec![false; n],
            };
            tmp_env.state.current_player_idx = target_seat;
            let mut mask = tmp_env.get_action_mask();
            mask.enable(Action::CheckCall);
            mask.enable(rec.action);
            if mask.is_valid(Action::BetSmall)
                || mask.is_valid(Action::BetPot33)
                || mask.is_valid(Action::BetPot75)
                || mask.is_valid(Action::AllIn)
            {
                snapshots.push((replay.clone(), mask, rec.action));
            }
        }

        let p = &mut replay.players[rec.player_idx];
        match rec.action {
            Action::Fold => {
                p.is_folded = true;
            }
            _ => {
                let spent = rec.amount.min(p.stack);
                p.stack -= spent;
                p.current_bet += spent;
                p.total_contributed += spent;
                replay.pot += spent;
                if p.stack == 0 {
                    p.is_all_in = true;
                }
                if p.current_bet > replay.current_highest_bet {
                    let diff = p.current_bet - replay.current_highest_bet;
                    if diff > replay.min_raise {
                        replay.min_raise = diff;
                    }
                    replay.current_highest_bet = p.current_bet;
                }
            }
        }
        replay.action_history.push(rec.clone());
    }

    if snapshots.len() > 4 {
        snapshots.drain(0..snapshots.len() - 4);
    }
    snapshots
}

/// Xây dựng range phân phối xác suất cho `target_seat`:
/// - Nếu bài đã biết -> `([c0, c1], [1.0], 0)`
/// - Nếu chưa biết -> sample tối đa `target_size` tay bài duy nhất và tính trọng số **Bayesian Likelihood**
///   trên `action_history` bằng Blueprint Net (`\prod_t max(\pi(a_t | s_t, h), 1e-3)`).
///   Nếu không có lịch sử hợp lệ -> trọng số đều hoàn toàn `1.0 / N` (0% hardcode).
fn build_bayesian_range<R: Rng>(
    env: &Environment,
    target_seat: usize,
    forced_hand0: Option<[Card; 2]>,
    prefer_avoid: Option<[Card; 2]>,
    target_size: usize,
    adv_net: &MLP,
    strat_net: Option<&MLP>,
    rng: &mut R,
) -> (Vec<[Card; 2]>, Vec<f32>, usize) {
    let target_p = &env.state.players[target_seat];
    if forced_hand0.is_none() {
        if let (Some(c0), Some(c1)) = (target_p.hole_cards[0], target_p.hole_cards[1]) {
            return (vec![[c0, c1]], vec![1.0], 0);
        }
    }

    let mut board_and_known = [false; 52];
    for c in &env.state.community_cards {
        board_and_known[c.0 as usize] = true;
    }
    for (i, p) in env.state.players.iter().enumerate() {
        if i != target_seat {
            if let (Some(c0), Some(c1)) = (p.hole_cards[0], p.hole_cards[1]) {
                // Giữ nguyên bài Hero trong bộ bài công khai khi đối thủ suy luận public range;
                // chỉ đánh dấu dead card đối với các bài thực sự công khai / đã ngửa.
                if Some([c0, c1]) != forced_hand0 && Some([c0, c1]) != prefer_avoid {
                    board_and_known[c0.0 as usize] = true;
                    board_and_known[c1.0 as usize] = true;
                }
            }
        }
    }

    let public_cards: Vec<Card> = (0..52)
        .filter(|&idx| !board_and_known[idx])
        .map(|idx| Card(idx as u8))
        .collect();

    if public_cards.len() < 2 {
        return (Vec::new(), Vec::new(), 0);
    }

    let mut hands: Vec<[Card; 2]> = Vec::with_capacity(target_size);
    let mut seen: HashSet<(u8, u8)> = HashSet::with_capacity(target_size * 2);

    // Nếu là Hero public range (cho đối thủ cập nhật CFR không nhìn trộm bài thật):
    // khe [0] luôn là bài thật của Hero.
    if let Some([hc0, hc1]) = forced_hand0 {
        let key = if hc0.0 <= hc1.0 {
            (hc0.0, hc1.0)
        } else {
            (hc1.0, hc0.0)
        };
        seen.insert(key);
        hands.push([hc0, hc1]);
    }

    let mut non_avoid_cards: Vec<Card> = if let Some([ac0, ac1]) = prefer_avoid {
        public_cards
            .iter()
            .copied()
            .filter(|&c| c.0 != ac0.0 && c.0 != ac1.0)
            .collect()
    } else {
        public_cards.clone()
    };
    if non_avoid_cards.len() < 2 {
        non_avoid_cards = public_cards.clone();
    }

    let max_attempts = target_size * 12;
    for _ in 0..max_attempts {
        if hands.len() >= target_size {
            break;
        }
        let pool = if prefer_avoid.is_some() && hands.len() * 4 < target_size * 3 && non_avoid_cards.len() >= 2 {
            &non_avoid_cards
        } else {
            &public_cards
        };
        let i0 = rng.gen_range(0..pool.len());
        let mut i1 = rng.gen_range(0..pool.len() - 1);
        if i1 >= i0 {
            i1 += 1;
        }
        let (a, b) = (pool[i0], pool[i1]);
        let key = if a.0 <= b.0 { (a.0, b.0) } else { (b.0, a.0) };
        if seen.insert(key) {
            hands.push([a, b]);
        }
    }

    if hands.is_empty() {
        return (Vec::new(), Vec::new(), 0);
    }

    let snapshots = extract_player_history_snapshots(env, target_seat);
    let hist_steps = snapshots.len();

    let mut weights = vec![1.0f32; hands.len()];
    if !snapshots.is_empty() {
        for (h_idx, &[c0, c1]) in hands.iter().enumerate() {
            let mut likelihood = 1.0f32;
            for (snap, mask, act) in &snapshots {
                let mut s_clone = snap.clone();
                s_clone.players[target_seat].hole_cards = [Some(c0), Some(c1)];
                let probs = blueprint_probs_for_state(&s_clone, target_seat, mask, adv_net, strat_net);
                let p_act = probs[act.to_index()].max(BAYESIAN_PROB_FLOOR);
                likelihood *= p_act;
            }
            weights[h_idx] = likelihood;
        }
    }

    let sum_w: f32 = weights.iter().sum();
    if sum_w > 0.0 && sum_w.is_finite() {
        for w in &mut weights {
            *w /= sum_w;
        }
    } else {
        let u = 1.0 / hands.len() as f32;
        weights.fill(u);
    }

    (hands, weights, hist_steps)
}

#[inline]
fn kish_ess(weights: &[f32]) -> f32 {
    let sum: f32 = weights.iter().sum();
    let sum_sq: f32 = weights.iter().map(|&w| w * w).sum();
    if sum_sq > 1e-12 {
        (sum * sum) / sum_sq
    } else {
        weights.len() as f32
    }
}

/// Terminal payoff tuple cho 2-sided CFR+:
/// `(h_win, h_tie, h_lose, v_win, v_tie, v_lose)`
#[derive(Clone, Copy)]
struct TwoPlayerPayoffs {
    h_win: f32,
    h_tie: f32,
    h_lose: f32,
    v_win: f32,
    v_tie: f32,
    v_lose: f32,
}

enum PublicNode {
    Terminal(TwoPlayerPayoffs),
    Decision {
        actor: usize, // 0 = Hero, 1 = Opponent
        actions: Vec<Action>,
        children: Vec<usize>,
        // Row-major: [hand_idx * num_actions + act_idx]
        regrets: Vec<f32>,
        policy_sum: Vec<f32>,
    },
}

fn eval_single_stack_outcome(
    stacks: &[f32],
    hero_seat: usize,
    opp_seat: usize,
    init_hero_stack: u32,
    init_opp_stack: u32,
    hero_start_icm: f32,
    opp_start_icm: f32,
    bb: f32,
    use_icm: bool,
) -> (f32, f32) {
    if use_icm {
        let payouts = get_sng_payouts(stacks.len());
        let evs = calculate_icm_ev(stacks, payouts);
        let h_ev = (evs.get(hero_seat).copied().unwrap_or(0.0) - hero_start_icm) * 10.0;
        let v_ev = (evs.get(opp_seat).copied().unwrap_or(0.0) - opp_start_icm) * 10.0;
        (h_ev, v_ev)
    } else {
        let h_ev = (stacks[hero_seat] - init_hero_stack as f32) / bb;
        let v_ev = (stacks[opp_seat] - init_opp_stack as f32) / bb;
        (h_ev, v_ev)
    }
}

fn compute_two_player_terminal_payoffs(
    cur_env: &Environment,
    root_env: &Environment,
    hero_seat: usize,
    opp_seat: usize,
    hero_start_icm: f32,
    opp_start_icm: f32,
    use_icm: bool,
) -> TwoPlayerPayoffs {
    let bb = root_env.state.bb_size.max(1) as f32;
    let init_h = root_env.state.players[hero_seat].stack;
    let init_v = root_env.state.players[opp_seat].stack;

    let mut base_stacks: Vec<f32> = cur_env.state.players.iter().map(|p| p.stack as f32).collect();
    let pot = cur_env.state.pot as f32;

    let hero_folded = cur_env.state.players[hero_seat].is_folded;
    let opp_folded = cur_env.state.players[opp_seat].is_folded;

    if hero_folded {
        base_stacks[opp_seat] += pot;
        let (he, ve) = eval_single_stack_outcome(
            &base_stacks,
            hero_seat,
            opp_seat,
            init_h,
            init_v,
            hero_start_icm,
            opp_start_icm,
            bb,
            use_icm,
        );
        return TwoPlayerPayoffs {
            h_win: he,
            h_tie: he,
            h_lose: he,
            v_win: ve,
            v_tie: ve,
            v_lose: ve,
        };
    }

    if opp_folded {
        base_stacks[hero_seat] += pot;
        let (he, ve) = eval_single_stack_outcome(
            &base_stacks,
            hero_seat,
            opp_seat,
            init_h,
            init_v,
            hero_start_icm,
            opp_start_icm,
            bb,
            use_icm,
        );
        return TwoPlayerPayoffs {
            h_win: he,
            h_tie: he,
            h_lose: he,
            v_win: ve,
            v_tie: ve,
            v_lose: ve,
        };
    }

    // Showdown hoặc street boundary: hoàn trả phần cược lệch chưa được call (side-pot / truncation)
    let commit_h = root_env.state.players[hero_seat].current_bet
        + init_h.saturating_sub(cur_env.state.players[hero_seat].stack);
    let commit_v = root_env.state.players[opp_seat].current_bet
        + init_v.saturating_sub(cur_env.state.players[opp_seat].stack);

    let mut contested_pot = pot;
    if commit_h > commit_v {
        let excess = ((commit_h - commit_v) as f32).min(contested_pot);
        base_stacks[hero_seat] += excess;
        contested_pot -= excess;
    } else if commit_v > commit_h {
        let excess = ((commit_v - commit_h) as f32).min(contested_pot);
        base_stacks[opp_seat] += excess;
        contested_pot -= excess;
    }

    let mut s_h_wins = base_stacks.clone();
    s_h_wins[hero_seat] += contested_pot;
    let (h_win, v_lose) = eval_single_stack_outcome(
        &s_h_wins,
        hero_seat,
        opp_seat,
        init_h,
        init_v,
        hero_start_icm,
        opp_start_icm,
        bb,
        use_icm,
    );

    let mut s_tie = base_stacks.clone();
    s_tie[hero_seat] += contested_pot * 0.5;
    s_tie[opp_seat] += contested_pot * 0.5;
    let (h_tie, v_tie) = eval_single_stack_outcome(
        &s_tie,
        hero_seat,
        opp_seat,
        init_h,
        init_v,
        hero_start_icm,
        opp_start_icm,
        bb,
        use_icm,
    );

    let mut s_v_wins = base_stacks;
    s_v_wins[opp_seat] += contested_pot;
    let (h_lose, v_win) = eval_single_stack_outcome(
        &s_v_wins,
        hero_seat,
        opp_seat,
        init_h,
        init_v,
        hero_start_icm,
        opp_start_icm,
        bb,
        use_icm,
    );

    TwoPlayerPayoffs {
        h_win,
        h_tie,
        h_lose,
        v_win,
        v_tie,
        v_lose,
    }
}

fn precompute_2player_matchups<R: Rng>(
    community: &[Card],
    h_hands: &[[Card; 2]],
    v_hands: &[[Card; 2]],
    rng: &mut R,
) -> (Vec<Vec<bool>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let nh = h_hands.len();
    let nv = v_hands.len();
    let mut compat = vec![vec![false; nv]; nh];
    let mut win_p = vec![vec![0.0f32; nv]; nh];
    let mut tie_p = vec![vec![0.0f32; nv]; nh];

    for i in 0..nh {
        let [ha, hb] = h_hands[i];
        for j in 0..nv {
            let [va, vb] = v_hands[j];
            compat[i][j] = ha.0 != va.0 && ha.0 != vb.0 && hb.0 != va.0 && hb.0 != vb.0;
        }
    }

    if community.len() >= 5 {
        let mut buf = [Card(0); 7];
        for (idx, &c) in community.iter().take(5).enumerate() {
            buf[idx] = c;
        }
        let h_ranks: Vec<HandRank> = h_hands
            .iter()
            .map(|&[c0, c1]| {
                buf[5] = c0;
                buf[6] = c1;
                evaluate_7_cards(&buf)
            })
            .collect();
        let v_ranks: Vec<HandRank> = v_hands
            .iter()
            .map(|&[c0, c1]| {
                buf[5] = c0;
                buf[6] = c1;
                evaluate_7_cards(&buf)
            })
            .collect();

        for i in 0..nh {
            for j in 0..nv {
                if compat[i][j] {
                    if h_ranks[i] > v_ranks[j] {
                        win_p[i][j] = 1.0;
                    } else if h_ranks[i] == v_ranks[j] {
                        tie_p[i][j] = 1.0;
                    }
                }
            }
        }
        return (compat, win_p, tie_p);
    }

    // Chưa đủ 5 lá bài chung: roll `HU_RUNOUT_SAMPLES` mặt bài chung ngẫu nhiên (0% hardcode)
    let mut board_mask = [false; 52];
    for c in community {
        board_mask[c.0 as usize] = true;
    }
    let rem_cards: Vec<Card> = (0..52)
        .filter(|&idx| !board_mask[idx])
        .map(|idx| Card(idx as u8))
        .collect();
    let needed = 5usize.saturating_sub(community.len());

    let mut runouts: Vec<Vec<Card>> = Vec::with_capacity(HU_RUNOUT_SAMPLES);
    if rem_cards.len() >= needed {
        for _ in 0..HU_RUNOUT_SAMPLES {
            let mut pool = rem_cards.clone();
            for k in 0..needed {
                let pick = k + rng.gen_range(0..pool.len() - k);
                pool.swap(k, pick);
            }
            let mut full_b = community.to_vec();
            full_b.extend_from_slice(&pool[..needed]);
            runouts.push(full_b);
        }
    }

    for i in 0..nh {
        let [ha, hb] = h_hands[i];
        for j in 0..nv {
            if !compat[i][j] {
                continue;
            }
            let [va, vb] = v_hands[j];
            let mut wins = 0.0f32;
            let mut ties = 0.0f32;
            let mut valid_r = 0usize;

            for b in &runouts {
                if b[community.len()..]
                    .iter()
                    .any(|&c| c.0 == ha.0 || c.0 == hb.0 || c.0 == va.0 || c.0 == vb.0)
                {
                    continue;
                }
                let mut h_cards = [Card(0); 7];
                let mut v_cards = [Card(0); 7];
                for k in 0..5 {
                    h_cards[k] = b[k];
                    v_cards[k] = b[k];
                }
                h_cards[5] = ha;
                h_cards[6] = hb;
                v_cards[5] = va;
                v_cards[6] = vb;
                let hr = evaluate_7_cards(&h_cards);
                let vr = evaluate_7_cards(&v_cards);
                if hr > vr {
                    wins += 1.0;
                } else if hr == vr {
                    ties += 1.0;
                }
                valid_r += 1;
            }

            if valid_r > 0 {
                win_p[i][j] = wins / valid_r as f32;
                tie_p[i][j] = ties / valid_r as f32;
            } else {
                win_p[i][j] = 0.5;
            }
        }
    }

    (compat, win_p, tie_p)
}

fn cfr_2sided_pass(
    nodes: &mut [PublicNode],
    node_idx: usize,
    reach_h: &[f32],
    reach_v: &[f32],
    compat: &[Vec<bool>],
    win_p: &[Vec<f32>],
    tie_p: &[Vec<f32>],
    iter_weight: f32,
) -> (Vec<f32>, Vec<f32>) {
    let nh = reach_h.len();
    let nv = reach_v.len();

    if let PublicNode::Terminal(p) = nodes[node_idx] {
        let mut cfv_h = vec![0.0f32; nh];
        let mut cfv_v = vec![0.0f32; nv];
        for i in 0..nh {
            let rhi = reach_h[i];
            let mut acc_h = 0.0f32;
            for j in 0..nv {
                if !compat[i][j] {
                    continue;
                }
                let wp = win_p[i][j];
                let tp = tie_p[i][j];
                let lp = (1.0 - wp - tp).max(0.0);
                let uh = wp * p.h_win + tp * p.h_tie + lp * p.h_lose;
                let uv = lp * p.v_win + tp * p.v_tie + wp * p.v_lose;
                acc_h += reach_v[j] * uh;
                cfv_v[j] += rhi * uv;
            }
            cfv_h[i] = acc_h;
        }
        return (cfv_h, cfv_v);
    }

    let (actor, na, children, strats) = match &nodes[node_idx] {
        PublicNode::Decision {
            actor,
            actions,
            children,
            regrets,
            ..
        } => {
            let n_hands = if *actor == 0 { nh } else { nv };
            let na = actions.len();
            let mut strats = vec![0.0f32; n_hands * na];
            for h in 0..n_hands {
                let off = h * na;
                let pos_sum: f32 = (0..na).map(|a| regrets[off + a].max(0.0)).sum();
                if pos_sum > 1e-9 {
                    for a in 0..na {
                        strats[off + a] = regrets[off + a].max(0.0) / pos_sum;
                    }
                } else {
                    let u = 1.0 / na as f32;
                    for a in 0..na {
                        strats[off + a] = u;
                    }
                }
            }
            (*actor, na, children.clone(), strats)
        }
        PublicNode::Terminal(_) => unreachable!(),
    };

    let mut child_h: Vec<Vec<f32>> = Vec::with_capacity(na);
    let mut child_v: Vec<Vec<f32>> = Vec::with_capacity(na);

    for a in 0..na {
        if actor == 0 {
            let mut next_rh = Vec::with_capacity(nh);
            for i in 0..nh {
                next_rh.push(reach_h[i] * strats[i * na + a]);
            }
            let (ch, cv) = cfr_2sided_pass(
                nodes,
                children[a],
                &next_rh,
                reach_v,
                compat,
                win_p,
                tie_p,
                iter_weight,
            );
            child_h.push(ch);
            child_v.push(cv);
        } else {
            let mut next_rv = Vec::with_capacity(nv);
            for j in 0..nv {
                next_rv.push(reach_v[j] * strats[j * na + a]);
            }
            let (ch, cv) = cfr_2sided_pass(
                nodes,
                children[a],
                reach_h,
                &next_rv,
                compat,
                win_p,
                tie_p,
                iter_weight,
            );
            child_h.push(ch);
            child_v.push(cv);
        }
    }

    if actor == 0 {
        let mut out_v = vec![0.0f32; nv];
        let mut out_h = vec![0.0f32; nh];
        for a in 0..na {
            for j in 0..nv {
                out_v[j] += child_v[a][j];
            }
            for i in 0..nh {
                out_h[i] += strats[i * na + a] * child_h[a][i];
            }
        }
        if let PublicNode::Decision {
            regrets,
            policy_sum,
            ..
        } = &mut nodes[node_idx]
        {
            for i in 0..nh {
                let off = i * na;
                let rh = reach_h[i].max(1e-6);
                for a in 0..na {
                    let s = strats[off + a];
                    regrets[off + a] = (regrets[off + a] + (child_h[a][i] - out_h[i])).max(0.0);
                    policy_sum[off + a] += iter_weight * rh * s;
                }
            }
        }
        (out_h, out_v)
    } else {
        let mut out_h = vec![0.0f32; nh];
        let mut out_v = vec![0.0f32; nv];
        for a in 0..na {
            for i in 0..nh {
                out_h[i] += child_h[a][i];
            }
            for j in 0..nv {
                out_v[j] += strats[j * na + a] * child_v[a][j];
            }
        }
        if let PublicNode::Decision {
            regrets,
            policy_sum,
            ..
        } = &mut nodes[node_idx]
        {
            for j in 0..nv {
                let off = j * na;
                let rv = reach_v[j].max(1e-6);
                for a in 0..na {
                    let s = strats[off + a];
                    regrets[off + a] = (regrets[off + a] + (child_v[a][j] - out_v[j])).max(0.0);
                    policy_sum[off + a] += iter_weight * rv * s;
                }
            }
        }
        (out_h, out_v)
    }
}

/// Đánh giá EV của cặp tay bài đơn `(hero_idx = 0, opp_idx = j)` dưới chiến thuật trung bình hội tụ `policy_sum`.
fn eval_avg_strategy_pair(
    nodes: &[PublicNode],
    node_idx: usize,
    hero_hand_idx: usize,
    opp_hand_idx: usize,
    wp: f32,
    tp: f32,
) -> f32 {
    match &nodes[node_idx] {
        PublicNode::Terminal(p) => {
            let lp = (1.0 - wp - tp).max(0.0);
            wp * p.h_win + tp * p.h_tie + lp * p.h_lose
        }
        PublicNode::Decision {
            actor,
            actions,
            children,
            policy_sum,
            regrets,
        } => {
            let na = actions.len();
            let h_idx = if *actor == 0 { hero_hand_idx } else { opp_hand_idx };
            let off = h_idx * na;
            let p_sum: f32 = (0..na).map(|a| policy_sum[off + a]).sum();

            let mut ev = 0.0f32;
            if p_sum > 1e-9 {
                for a in 0..na {
                    let prob = policy_sum[off + a] / p_sum;
                    if prob > 1e-6 {
                        ev += prob
                            * eval_avg_strategy_pair(
                                nodes,
                                children[a],
                                hero_hand_idx,
                                opp_hand_idx,
                                wp,
                                tp,
                            );
                    }
                }
            } else {
                let r_sum: f32 = (0..na).map(|a| regrets[off + a].max(0.0)).sum();
                for a in 0..na {
                    let prob = if r_sum > 1e-9 {
                        regrets[off + a].max(0.0) / r_sum
                    } else {
                        1.0 / na as f32
                    };
                    if prob > 1e-6 {
                        ev += prob
                            * eval_avg_strategy_pair(
                                nodes,
                                children[a],
                                hero_hand_idx,
                                opp_hand_idx,
                                wp,
                                tp,
                            );
                    }
                }
            }
            ev
        }
    }
}

#[derive(Clone)]
struct MultiwayNode {
    regrets: [f32; NUM_ACTIONS],
    policy_sum: [f32; NUM_ACTIONS],
}

impl MultiwayNode {
    fn new(prior: &[f32; NUM_ACTIONS], mask: &ActionMask) -> Self {
        let mut regrets = [0.0f32; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            if mask.is_index_valid(a) {
                regrets[a] = prior[a] * 0.05;
            }
        }
        Self {
            regrets,
            policy_sum: [0.0f32; NUM_ACTIONS],
        }
    }

    fn current_strategy(&self, mask: &ActionMask) -> [f32; NUM_ACTIONS] {
        DeepCFRPolicy::compute_regret_matching_probs(&self.regrets, mask)
    }

    fn average_strategy(&self, mask: &ActionMask) -> [f32; NUM_ACTIONS] {
        let mut out = [0.0f32; NUM_ACTIONS];
        let valid = mask.valid_actions();
        let sum: f32 = valid.iter().map(|a| self.policy_sum[a.to_index()].max(0.0)).sum();
        if sum > 1e-6 {
            for a in valid {
                out[a.to_index()] = self.policy_sum[a.to_index()].max(0.0) / sum;
            }
            out
        } else {
            self.current_strategy(mask)
        }
    }
}

#[inline]
fn fast_hand_bucket(
    hole: [Card; 2],
    community: &[Card],
    opp_count: usize,
) -> u8 {
    if community.len() >= 3 {
        let mut cards = Vec::with_capacity(community.len() + 2);
        cards.extend_from_slice(community);
        cards.push(hole[0]);
        cards.push(hole[1]);
        let rank = evaluate_7_cards(&cards);
        ((rank.0 >> 24) as u8).min(9)
    } else {
        let eq = preflop_table().get(hole[0], hole[1], opp_count.max(1));
        ((eq * 10.0).floor() as i32).clamp(0, 9) as u8
    }
}

#[inline]
fn hash_seq(seq_hash: u64, act_idx: usize) -> u64 {
    seq_hash
        .wrapping_mul(1_099_511_628_211)
        .wrapping_add((act_idx as u64) + 1)
}

pub struct SubgameSolver {
    pub config: SubgameSolverConfig,
}

impl SubgameSolver {
    pub fn new(config: SubgameSolverConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: SubgameSolverConfig::default(),
        }
    }

    /// Bộ lọc kích hoạt: Kiểm tra xem ván bài hiện tại có nên chạy Depth-Limited Search hay không
    pub fn should_trigger_search(
        &self,
        env: &Environment,
        hero_seat: usize,
        force_search: bool,
    ) -> SearchTrigger {
        if force_search {
            return SearchTrigger::ManualForced;
        }

        let num_alive = env
            .state
            .players
            .iter()
            .filter(|p| p.stack > 0 || p.is_all_in)
            .count();
        if num_alive <= 2 {
            return SearchTrigger::HeadsUpDuel;
        }

        let bb = env.state.bb_size.max(1) as f32;
        let pot_bb = env.state.pot as f32 / bb;

        if pot_bb >= self.config.min_pot_bb_trigger {
            return SearchTrigger::HighStakesPot { pot_bb };
        }

        let hero_bet = env.state.players[hero_seat].current_bet;
        let to_call_u32 = env.state.current_highest_bet.saturating_sub(hero_bet);
        let to_call = to_call_u32 as f32;
        let pot_before_bet = env.state.pot.saturating_sub(to_call_u32) as f32;
        if pot_before_bet > 0.0
            && to_call > bb
            && (env.state.street != Street::Preflop || to_call >= 4.0 * bb)
            && (to_call / pot_before_bet) >= self.config.unusual_bet_ratio_trigger
        {
            return SearchTrigger::UnusualOverbet {
                bet_pot_ratio: to_call / pot_before_bet,
            };
        }

        let hero_stack = env.state.players[hero_seat].stack as f32;
        if hero_stack > 0.0 && (to_call / hero_stack) >= 0.35 {
            return SearchTrigger::DeepStackCommitment {
                to_call_ratio: to_call / hero_stack,
            };
        }

        if self.config.search_turn_river
            && (env.state.street == Street::Turn || env.state.street == Street::River)
        {
            return SearchTrigger::CrucialPostflopStreet {
                street: env.state.street,
            };
        }

        SearchTrigger::FastPolicyFallback
    }

    #[inline]
    fn evaluate_resolved_stacks(
        &self,
        env: &Environment,
        target_seat: usize,
        initial_target_stack: u32,
        target_start_icm: f32,
    ) -> f32 {
        if self.config.use_icm {
            let end_stacks: Vec<f32> = env.state.players.iter().map(|p| p.stack as f32).collect();
            let payouts = get_sng_payouts(end_stacks.len());
            let end_evs = calculate_icm_ev(&end_stacks, payouts);
            let end_icm = end_evs.get(target_seat).copied().unwrap_or(0.0);
            (end_icm - target_start_icm) * 10.0
        } else {
            let bb = env.state.bb_size.max(1) as f32;
            let final_stack = env.state.players[target_seat].stack as f32;
            (final_stack - initial_target_stack as f32) / bb
        }
    }

    fn resolve_pot_into_stacks(sim: &mut Environment) {
        if sim.state.pot == 0 {
            return;
        }
        let in_hand: Vec<usize> = sim
            .state
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_in_hand())
            .map(|(i, _)| i)
            .collect();
        if in_hand.is_empty() {
            return;
        }
        if in_hand.len() == 1 {
            sim.state.players[in_hand[0]].stack += sim.state.pot;
            sim.state.pot = 0;
            return;
        }

        while sim.state.community_cards.len() < 5 {
            match sim.deck.deal() {
                Some(c) => sim.state.community_cards.push(c),
                None => break,
            }
        }

        let all_have_hole = in_hand
            .iter()
            .all(|&i| sim.state.players[i].hole_cards[0].is_some() && sim.state.players[i].hole_cards[1].is_some());

        if all_have_hole && sim.state.community_cards.len() == 5 {
            let all_have_contrib = in_hand.iter().all(|&i| sim.state.players[i].total_contributed > 0);
            if all_have_contrib {
                sim.resolve_showdown();
                return;
            }

            let mut best_score: Option<HandRank> = None;
            let mut winners = Vec::new();
            for &idx in &in_hand {
                let p = &sim.state.players[idx];
                let mut cards = sim.state.community_cards.clone();
                cards.push(p.hole_cards[0].unwrap());
                cards.push(p.hole_cards[1].unwrap());
                let rank = evaluate_7_cards(&cards);
                match best_score {
                    None => {
                        best_score = Some(rank);
                        winners.push(idx);
                    }
                    Some(br) if rank > br => {
                        best_score = Some(rank);
                        winners.clear();
                        winners.push(idx);
                    }
                    Some(br) if rank == br => {
                        winners.push(idx);
                    }
                    _ => {}
                }
            }
            if !winners.is_empty() {
                let share = sim.state.pot / winners.len() as u32;
                let rem = sim.state.pot % winners.len() as u32;
                for (pos, &w) in winners.iter().enumerate() {
                    let bonus = if pos < rem as usize { 1 } else { 0 };
                    sim.state.players[w].stack += share + bonus;
                }
                sim.state.pot = 0;
                return;
            }
        }

        let share = sim.state.pot / in_hand.len() as u32;
        let rem = sim.state.pot % in_hand.len() as u32;
        for (pos, &idx) in in_hand.iter().enumerate() {
            let bonus = if pos < rem as usize { 1 } else { 0 };
            sim.state.players[idx].stack += share + bonus;
        }
        sim.state.pot = 0;
    }

    #[inline]
    fn evaluate_multiway_outcome(
        &self,
        env: &Environment,
        target_seat: usize,
        initial_target_stack: u32,
        target_start_icm: f32,
    ) -> f32 {
        if env.state.pot == 0 {
            return self.evaluate_resolved_stacks(env, target_seat, initial_target_stack, target_start_icm);
        }

        let target_player = &env.state.players[target_seat];
        if target_player.is_folded {
            if !self.config.use_icm {
                let bb = env.state.bb_size.max(1) as f32;
                return (target_player.stack as f32 - initial_target_stack as f32) / bb;
            }
            let mut sim = env.clone();
            Self::resolve_pot_into_stacks(&mut sim);
            return self.evaluate_resolved_stacks(&sim, target_seat, initial_target_stack, target_start_icm);
        }

        let in_hand: Vec<usize> = env
            .state
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_in_hand())
            .map(|(i, _)| i)
            .collect();
        let in_hand_count = in_hand.len();

        let all_have_hole = in_hand
            .iter()
            .all(|&i| env.state.players[i].hole_cards[0].is_some() && env.state.players[i].hole_cards[1].is_some());
        let can_finish_board = env.state.community_cards.len() + env.deck.remaining() >= 5;

        if in_hand_count <= 1 || (all_have_hole && can_finish_board) {
            let mut sim = env.clone();
            Self::resolve_pot_into_stacks(&mut sim);
            return self.evaluate_resolved_stacks(&sim, target_seat, initial_target_stack, target_start_icm);
        }

        // Fallback equity vs random khi bộ bài không đủ chia thêm
        if self.config.use_icm {
            let mut end_stacks: Vec<f32> = env.state.players.iter().map(|p| p.stack as f32).collect();
            let pot = env.state.pot as f32;
            let eq = if let (Some(c0), Some(c1)) = (target_player.hole_cards[0], target_player.hole_cards[1]) {
                equity_vs_random([c0, c1], &env.state.community_cards, in_hand_count.saturating_sub(1))
            } else {
                1.0 / in_hand_count.max(1) as f32
            };
            end_stacks[target_seat] += eq * pot;
            let opp_count = in_hand_count.saturating_sub(1);
            if opp_count > 0 {
                let opp_share = ((1.0 - eq) * pot) / opp_count as f32;
                for &i in &in_hand {
                    if i != target_seat {
                        end_stacks[i] += opp_share;
                    }
                }
            }
            let payouts = get_sng_payouts(end_stacks.len());
            let end_evs = calculate_icm_ev(&end_stacks, payouts);
            let end_icm = end_evs.get(target_seat).copied().unwrap_or(0.0);
            (end_icm - target_start_icm) * 10.0
        } else {
            let bb = env.state.bb_size.max(1) as f32;
            let eq = if let (Some(c0), Some(c1)) = (target_player.hole_cards[0], target_player.hole_cards[1]) {
                equity_vs_random([c0, c1], &env.state.community_cards, in_hand_count.saturating_sub(1))
            } else {
                1.0 / in_hand_count.max(1) as f32
            };
            let expected_chips = target_player.stack as f32 + (eq * env.state.pot as f32);
            (expected_chips - initial_target_stack as f32) / bb
        }
    }

    fn build_public_tree_node(
        &self,
        env: &Environment,
        root_env: &Environment,
        root_street: Street,
        depth: usize,
        hero_seat: usize,
        opp_seat: usize,
        hero_start_icm: f32,
        opp_start_icm: f32,
        root_mask: Option<ActionMask>,
        nh: usize,
        nv: usize,
        nodes: &mut Vec<PublicNode>,
    ) -> usize {
        let in_hand_count = env.state.players.iter().filter(|p| p.is_in_hand()).count();
        if env.state.street != root_street
            || env.state.street == Street::Showdown
            || in_hand_count <= 1
            || depth >= self.config.max_depth
            || nodes.len() >= HU_MAX_TREE_NODES
        {
            let payoffs = compute_two_player_terminal_payoffs(
                env,
                root_env,
                hero_seat,
                opp_seat,
                hero_start_icm,
                opp_start_icm,
                self.config.use_icm,
            );
            let idx = nodes.len();
            nodes.push(PublicNode::Terminal(payoffs));
            return idx;
        }

        let cur = env.state.current_player_idx;
        if cur != hero_seat && cur != opp_seat {
            let payoffs = compute_two_player_terminal_payoffs(
                env,
                root_env,
                hero_seat,
                opp_seat,
                hero_start_icm,
                opp_start_icm,
                self.config.use_icm,
            );
            let idx = nodes.len();
            nodes.push(PublicNode::Terminal(payoffs));
            return idx;
        }

        let mask = root_mask.unwrap_or_else(|| env.get_action_mask());
        let valid_actions = mask.valid_actions();
        if valid_actions.is_empty() {
            let payoffs = compute_two_player_terminal_payoffs(
                env,
                root_env,
                hero_seat,
                opp_seat,
                hero_start_icm,
                opp_start_icm,
                self.config.use_icm,
            );
            let idx = nodes.len();
            nodes.push(PublicNode::Terminal(payoffs));
            return idx;
        }

        let actor = if cur == hero_seat { 0 } else { 1 };
        let n_hands = if actor == 0 { nh } else { nv };
        let na = valid_actions.len();

        let node_idx = nodes.len();
        nodes.push(PublicNode::Decision {
            actor,
            actions: valid_actions.clone(),
            children: Vec::with_capacity(na),
            regrets: vec![0.0; n_hands * na],
            policy_sum: vec![0.0; n_hands * na],
        });

        let mut children = Vec::with_capacity(na);
        for &act in &valid_actions {
            if nodes.len() >= HU_MAX_TREE_NODES {
                let mut tmp = env.clone();
                let _ = tmp.step(act);
                let payoffs = compute_two_player_terminal_payoffs(
                    &tmp,
                    root_env,
                    hero_seat,
                    opp_seat,
                    hero_start_icm,
                    opp_start_icm,
                    self.config.use_icm,
                );
                let tidx = nodes.len();
                nodes.push(PublicNode::Terminal(payoffs));
                children.push(tidx);
                continue;
            }

            let mut branch = env.clone();
            let done = branch.step(act).unwrap_or(true);
            let child_idx = if done
                || branch.state.street != root_street
                || branch.state.street == Street::Showdown
                || depth + 1 >= self.config.max_depth
            {
                let payoffs = compute_two_player_terminal_payoffs(
                    &branch,
                    root_env,
                    hero_seat,
                    opp_seat,
                    hero_start_icm,
                    opp_start_icm,
                    self.config.use_icm,
                );
                let tidx = nodes.len();
                nodes.push(PublicNode::Terminal(payoffs));
                tidx
            } else {
                self.build_public_tree_node(
                    &branch,
                    root_env,
                    root_street,
                    depth + 1,
                    hero_seat,
                    opp_seat,
                    hero_start_icm,
                    opp_start_icm,
                    None,
                    nh,
                    nv,
                    nodes,
                )
            };
            children.push(child_idx);
        }

        if let PublicNode::Decision {
            children: ref mut c_ref,
            ..
        } = nodes[node_idx]
        {
            *c_ref = children;
        }
        node_idx
    }

    /// Giải chính xác **2-Sided Range-vs-Range Linear CFR+** khi trong ván còn đúng 2 người chơi (1v1 hoặc multiway fold xuống 2).
    fn solve_heads_up_2sided_cfr(
        &self,
        env: &Environment,
        hero_seat: usize,
        opp_seat: usize,
        adv_net: &MLP,
        strat_net: Option<&MLP>,
        trigger: SearchTrigger,
        mask: ActionMask,
        start_time: Instant,
    ) -> SubgameResult {
        let mut rng = thread_rng();
        let hero_hole = match (
            env.state.players[hero_seat].hole_cards[0],
            env.state.players[hero_seat].hole_cards[1],
        ) {
            (Some(c0), Some(c1)) => Some([c0, c1]),
            _ => None,
        };

        let (h_hands, prior_h, h_steps) = build_bayesian_range(
            env,
            hero_seat,
            hero_hole,
            None,
            HU_RANGE_SIZE,
            adv_net,
            strat_net,
            &mut rng,
        );
        let (v_hands, prior_v, v_steps) = build_bayesian_range(
            env,
            opp_seat,
            None,
            hero_hole,
            HU_RANGE_SIZE,
            adv_net,
            strat_net,
            &mut rng,
        );

        // Fallback an toàn nếu không dựng được range công khai (deck rỗng)
        if h_hands.is_empty() || v_hands.is_empty() {
            let valid = mask.valid_actions();
            let act = valid.first().copied().unwrap_or(Action::CheckCall);
            let mut probs = [0.0f32; NUM_ACTIONS];
            probs[act.to_index()] = 1.0;
            return SubgameResult {
                best_action: act,
                action_probs: probs,
                action_evs: [0.0f32; NUM_ACTIONS],
                trigger,
                iterations_run: 0,
                duration_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                diagnostics: SubgameDiagnostics {
                    solver_kind: "2-Sided Fallback",
                    num_in_hand: 2,
                    ..Default::default()
                },
            };
        }

        let nh = h_hands.len();
        let nv = v_hands.len();
        let (compat, win_p, tie_p) =
            precompute_2player_matchups(&env.state.community_cards, &h_hands, &v_hands, &mut rng);

        // Xóa bài tẩy, bài chung & deck trên `public_env` khi dựng cây công khai
        // để `resolve_showdown` luôn thấy `cards.len() < 5` và giữ nguyên `pot`/`stack` cho 2-sided payoff.
        let mut public_env = env.clone();
        public_env.deck = Deck::from_remaining(&[]);
        public_env.state.community_cards.clear();
        for p in &mut public_env.state.players {
            p.hole_cards = [None, None];
        }

        let start_stacks: Vec<f32> = env.state.players.iter().map(|p| p.stack as f32).collect();
        let payouts = get_sng_payouts(start_stacks.len());
        let (hero_start_icm, opp_start_icm) = if self.config.use_icm && start_stacks.len() >= 2 {
            let evs = calculate_icm_ev(&start_stacks, payouts);
            (
                evs.get(hero_seat).copied().unwrap_or(0.0),
                evs.get(opp_seat).copied().unwrap_or(0.0),
            )
        } else {
            (0.0, 0.0)
        };

        let mut nodes = Vec::with_capacity(128);
        let root_idx = self.build_public_tree_node(
            &public_env,
            env,
            env.state.street,
            0,
            hero_seat,
            opp_seat,
            hero_start_icm,
            opp_start_icm,
            Some(mask),
            nh,
            nv,
            &mut nodes,
        );

        let iters = self.config.iterations.max(1);
        for t in 1..=iters {
            cfr_2sided_pass(
                &mut nodes,
                root_idx,
                &prior_h,
                &prior_v,
                &compat,
                &win_p,
                &tie_p,
                t as f32,
            );
        }

        // Trích xuất phân phối GTO, EV, ±SE và phản ứng tức thì của đối thủ cho tay bài thật của Hero (index 0)
        let mut action_evs = [f32::NEG_INFINITY; NUM_ACTIONS];
        let mut action_probs = [0.0f32; NUM_ACTIONS];
        let mut action_ev_se = [0.0f32; NUM_ACTIONS];
        let mut opp_fold_pct = [0.0f32; NUM_ACTIONS];
        let mut opp_call_pct = [0.0f32; NUM_ACTIONS];
        let mut opp_raise_pct = [0.0f32; NUM_ACTIONS];

        let ess = kish_ess(&prior_v);

        let mut compat_weights: Vec<(usize, f32)> = (0..nv)
            .filter(|&j| compat[0][j])
            .map(|j| (j, prior_v[j]))
            .collect();
        if compat_weights.is_empty() {
            for j in 0..nv {
                compat_weights.push((j, prior_v[j]));
            }
        }
        let norm_v: f32 = compat_weights.iter().map(|&(_, w)| w).sum::<f32>().max(1e-9);
        let compat_ess = {
            let s_sq: f32 = compat_weights.iter().map(|&(_, w)| w * w).sum();
            if s_sq > 1e-12 {
                (norm_v * norm_v) / s_sq
            } else {
                compat_weights.len() as f32
            }
        };

        let mut hero_showdown_eq = 0.0f32;
        for &(j, w) in &compat_weights {
            hero_showdown_eq += w * (win_p[0][j] + 0.5 * tie_p[0][j]);
        }
        hero_showdown_eq /= norm_v;

        if let PublicNode::Decision {
            actions,
            children,
            policy_sum,
            regrets,
            ..
        } = &nodes[root_idx]
        {
            let na = actions.len();
            let p_sum: f32 = (0..na).map(|a| policy_sum[a]).sum();
            if p_sum > 1e-9 {
                for (a_slot, &act) in actions.iter().enumerate() {
                    action_probs[act.to_index()] = policy_sum[a_slot] / p_sum;
                }
            } else {
                let r_sum: f32 = (0..na).map(|a| regrets[a].max(0.0)).sum();
                for (a_slot, &act) in actions.iter().enumerate() {
                    action_probs[act.to_index()] = if r_sum > 1e-9 {
                        regrets[a_slot].max(0.0) / r_sum
                    } else {
                        1.0 / na as f32
                    };
                }
            }

            for (a_slot, &act) in actions.iter().enumerate() {
                let act_i = act.to_index();
                let child_idx = children[a_slot];

                let mut pair_evs = Vec::with_capacity(compat_weights.len());
                let mut ev_sum = 0.0f32;
                for &(j, w) in &compat_weights {
                    let ev_j = eval_avg_strategy_pair(&nodes, child_idx, 0, j, win_p[0][j], tie_p[0][j]);
                    pair_evs.push((ev_j, w));
                    ev_sum += w * ev_j;
                }
                let mean_ev = ev_sum / norm_v;
                action_evs[act_i] = mean_ev;

                let mut var_sum = 0.0f32;
                for &(ev_j, w) in &pair_evs {
                    let d = ev_j - mean_ev;
                    var_sum += w * d * d;
                }
                let var = (var_sum / norm_v).max(0.0);
                action_ev_se[act_i] = (var / compat_ess.max(1.0)).sqrt();

                // Tính tần suất đối thủ Fold/Call/Raise trực tiếp sau hành động này của Hero
                if let PublicNode::Decision {
                    actor: opp_actor,
                    actions: opp_acts,
                    policy_sum: opp_psum,
                    regrets: opp_regs,
                    ..
                } = &nodes[child_idx]
                {
                    if *opp_actor == 1 {
                        let vna = opp_acts.len();
                        let mut f_acc = 0.0f32;
                        let mut c_acc = 0.0f32;
                        let mut r_acc = 0.0f32;
                        for &(j, w) in &compat_weights {
                            let off = j * vna;
                            let tot_p: f32 = (0..vna).map(|va| opp_psum[off + va]).sum();
                            let tot_r: f32 = (0..vna).map(|va| opp_regs[off + va].max(0.0)).sum();
                            for (va_slot, &va) in opp_acts.iter().enumerate() {
                                let p = if tot_p > 1e-9 {
                                    opp_psum[off + va_slot] / tot_p
                                } else if tot_r > 1e-9 {
                                    opp_regs[off + va_slot].max(0.0) / tot_r
                                } else {
                                    1.0 / vna as f32
                                };
                                match va {
                                    Action::Fold => f_acc += w * p,
                                    Action::CheckCall => c_acc += w * p,
                                    _ => r_acc += w * p,
                                }
                            }
                        }
                        opp_fold_pct[act_i] = f_acc / norm_v;
                        opp_call_pct[act_i] = c_acc / norm_v;
                        opp_raise_pct[act_i] = r_acc / norm_v;
                    }
                }
            }
        }

        let valid_actions = mask.valid_actions();
        let tie_eps = if self.config.use_icm { 1e-3 } else { 1e-2 };
        let mut best_action = valid_actions[0];
        let mut best_ev = action_evs[best_action.to_index()];
        let mut best_p = action_probs[best_action.to_index()];
        for &a in &valid_actions[1..] {
            let idx = a.to_index();
            let ev = action_evs[idx];
            let p = action_probs[idx];
            if ev > best_ev + tie_eps || ((ev - best_ev).abs() <= tie_eps && p > best_p) {
                best_ev = ev;
                best_p = p;
                best_action = a;
            }
        }

        let opp_can_act = env.state.players[opp_seat].is_active();
        let hero_is_ip = !opp_can_act || env.street_has_acted[opp_seat];

        SubgameResult {
            best_action,
            action_probs,
            action_evs,
            trigger,
            iterations_run: iters,
            duration_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            diagnostics: SubgameDiagnostics {
                solver_kind: "2-Sided Exact CFR+",
                num_in_hand: 2,
                hero_is_ip,
                opp_range_size: nv,
                opp_bayesian_ess: ess,
                bayesian_history_steps: v_steps + h_steps,
                hero_showdown_eq,
                action_ev_se,
                opp_fold_pct,
                opp_call_pct,
                opp_raise_pct,
            },
        }
    }

    fn traverse_multiway_cfr<R: Rng>(
        &self,
        env: &mut Environment,
        depth: usize,
        iter_weight: f32,
        traverser_seat: usize,
        initial_traverser_stack: u32,
        traverser_start_icm: f32,
        seq_hash: u64,
        adv_net: &MLP,
        strat_net: Option<&MLP>,
        infoset_table: &mut HashMap<(u8, u8, u64), MultiwayNode>,
        rng: &mut R,
    ) -> f32 {
        if env.state.street == Street::Showdown {
            return self.evaluate_multiway_outcome(
                env,
                traverser_seat,
                initial_traverser_stack,
                traverser_start_icm,
            );
        }

        let in_hand_count = env.state.players.iter().filter(|p| p.is_in_hand()).count();
        if in_hand_count <= 1 {
            return self.evaluate_multiway_outcome(
                env,
                traverser_seat,
                initial_traverser_stack,
                traverser_start_icm,
            );
        }

        if depth >= self.config.max_depth {
            return self.evaluate_multiway_outcome(
                env,
                traverser_seat,
                initial_traverser_stack,
                traverser_start_icm,
            );
        }

        let cur = env.state.current_player_idx;
        let mask = env.get_action_mask();
        if mask.0 == 0 {
            return self.evaluate_multiway_outcome(
                env,
                traverser_seat,
                initial_traverser_stack,
                traverser_start_icm,
            );
        }

        let cur_hole = match (
            env.state.players[cur].hole_cards[0],
            env.state.players[cur].hole_cards[1],
        ) {
            (Some(c0), Some(c1)) => [c0, c1],
            _ => [Card(0), Card(1)],
        };
        let bucket = fast_hand_bucket(
            cur_hole,
            &env.state.community_cards,
            in_hand_count.saturating_sub(1),
        );
        let key = (cur as u8, bucket, seq_hash);

        if !infoset_table.contains_key(&key) {
            let prior = blueprint_probs_for_state(&env.state, cur, &mask, adv_net, strat_net);
            infoset_table.insert(key, MultiwayNode::new(&prior, &mask));
        }

        let strat = infoset_table[&key].current_strategy(&mask);

        if cur == traverser_seat {
            let valid_actions = mask.valid_actions();
            let mut action_vals = [0.0f32; NUM_ACTIONS];
            let mut expected_v = 0.0f32;

            for &a in &valid_actions {
                let a_idx = a.to_index();
                let mut branch = env.clone();
                let done = branch.step(a).unwrap_or(true);
                let v = if done {
                    self.evaluate_multiway_outcome(
                        &branch,
                        traverser_seat,
                        initial_traverser_stack,
                        traverser_start_icm,
                    )
                } else {
                    self.traverse_multiway_cfr(
                        &mut branch,
                        depth + 1,
                        iter_weight,
                        traverser_seat,
                        initial_traverser_stack,
                        traverser_start_icm,
                        hash_seq(seq_hash, a_idx),
                        adv_net,
                        strat_net,
                        infoset_table,
                        rng,
                    )
                };
                action_vals[a_idx] = v;
                expected_v += strat[a_idx] * v;
            }

            if let Some(node) = infoset_table.get_mut(&key) {
                for &a in &valid_actions {
                    let a_idx = a.to_index();
                    node.regrets[a_idx] = (node.regrets[a_idx] + (action_vals[a_idx] - expected_v)).max(0.0);
                    node.policy_sum[a_idx] += iter_weight * strat[a_idx];
                }
            }
            expected_v
        } else {
            let a = sample_action_from_probs(&strat, &mask, rng);
            let a_idx = a.to_index();
            let mut branch = env.clone();
            let done = branch.step(a).unwrap_or(true);
            if done {
                self.evaluate_multiway_outcome(
                    &branch,
                    traverser_seat,
                    initial_traverser_stack,
                    traverser_start_icm,
                )
            } else {
                self.traverse_multiway_cfr(
                    &mut branch,
                    depth + 1,
                    iter_weight,
                    traverser_seat,
                    initial_traverser_stack,
                    traverser_start_icm,
                    hash_seq(seq_hash, a_idx),
                    adv_net,
                    strat_net,
                    infoset_table,
                    rng,
                )
            }
        }
    }

    /// Giải **N-Player Multiway External-Sampling CFR+** khi còn >= 3 người trong ván.
    fn solve_multiway_cfr(
        &self,
        env: &Environment,
        hero_seat: usize,
        in_hand_seats: &[usize],
        adv_net: &MLP,
        strat_net: Option<&MLP>,
        trigger: SearchTrigger,
        mask: ActionMask,
        start_time: Instant,
    ) -> SubgameResult {
        let mut rng = thread_rng();
        let valid_actions = mask.valid_actions();
        let hero_hole = match (
            env.state.players[hero_seat].hole_cards[0],
            env.state.players[hero_seat].hole_cards[1],
        ) {
            (Some(c0), Some(c1)) => Some([c0, c1]),
            _ => None,
        };

        let mut ranges: HashMap<usize, (Vec<[Card; 2]>, Vec<f32>, usize)> = HashMap::new();
        let mut total_hist_steps = 0usize;
        let mut opp_ess_sum = 0.0f32;
        let mut opp_ess_cnt = 0usize;
        let mut opp_range_size_total = 0usize;

        for &seat in in_hand_seats {
            let (hands, weights, steps) = build_bayesian_range(
                env,
                seat,
                if seat == hero_seat { hero_hole } else { None },
                if seat != hero_seat { hero_hole } else { None },
                MULTIWAY_RANGE_SIZE,
                adv_net,
                strat_net,
                &mut rng,
            );
            total_hist_steps += steps;
            if seat != hero_seat {
                opp_ess_sum += kish_ess(&weights);
                opp_ess_cnt += 1;
                opp_range_size_total = opp_range_size_total.max(hands.len());
            }
            ranges.insert(seat, (hands, weights, steps));
        }

        // Xây dựng tập bài công khai loại trừ bài chung và bài đã biết cố định (nếu có)
        let mut fixed_dead = [false; 52];
        for c in &env.state.community_cards {
            fixed_dead[c.0 as usize] = true;
        }
        if let Some([hc0, hc1]) = hero_hole {
            fixed_dead[hc0.0 as usize] = true;
            fixed_dead[hc1.0 as usize] = true;
        }

        let hero_initial_stack = env.state.players[hero_seat].stack;
        let start_stacks: Vec<f32> = env
            .state
            .players
            .iter()
            .map(|p| (p.stack + p.current_bet) as f32)
            .collect();
        let payouts = get_sng_payouts(start_stacks.len());
        let start_icms: Vec<f32> = if self.config.use_icm && start_stacks.len() >= 2 {
            calculate_icm_ev(&start_stacks, payouts)
        } else {
            vec![0.0; start_stacks.len()]
        };

        let mut infoset_table: HashMap<(u8, u8, u64), MultiwayNode> = HashMap::with_capacity(256);
        let mut sum_v = [0.0f64; NUM_ACTIONS];
        let mut sum_sq_v = [0.0f64; NUM_ACTIONS];
        let mut root_regrets = [0.0f32; NUM_ACTIONS];
        let mut root_policy_sum = [0.0f32; NUM_ACTIONS];

        let mut opp_resp_total = [0u32; NUM_ACTIONS];
        let mut opp_fold_cnt = [0u32; NUM_ACTIONS];
        let mut opp_call_cnt = [0u32; NUM_ACTIONS];
        let mut opp_raise_cnt = [0u32; NUM_ACTIONS];

        let root_features = env.state.encode_features(hero_seat);
        let warm_probs = if let Some(strat_mlp) = strat_net {
            let raw = strat_mlp.forward(&root_features);
            DeepCFRPolicy::compute_strategy_probs(&raw, &mask)
        } else {
            let raw = adv_net.forward(&root_features);
            DeepCFRPolicy::compute_regret_matching_probs(&raw, &mask)
        };
        for &a in &valid_actions {
            let i = a.to_index();
            root_regrets[i] = warm_probs[i] * 0.05;
        }

        let opp_seats: Vec<usize> = in_hand_seats
            .iter()
            .copied()
            .filter(|&s| s != hero_seat)
            .collect();

        // Tính Hero Showdown Equity ước lượng vs đối thủ trên sample
        let hero_showdown_eq = if let Some([hc0, hc1]) = hero_hole {
            equity_vs_random([hc0, hc1], &env.state.community_cards, opp_seats.len())
        } else {
            1.0 / in_hand_seats.len().max(1) as f32
        };

        let iters = self.config.iterations.max(1);
        for iter_idx in 0..iters {
            let iter_w = (iter_idx + 1) as f32;

            // Bước 1: Chọn 1 đối thủ cập nhật regret CFR+ (dùng Hero public range -> không nhìn trộm bài Hero!)
            if !opp_seats.is_empty() {
                let opp_target = opp_seats[iter_idx % opp_seats.len()];
                let mut opp_branch = env.clone();
                let mut used = [false; 52];
                for c in &env.state.community_cards {
                    used[c.0 as usize] = true;
                }
                for (i, p) in opp_branch.state.players.iter().enumerate() {
                    if i != hero_seat {
                        for c in p.hole_cards.iter().flatten() {
                            used[c.0 as usize] = true;
                        }
                    }
                }
                // Gán bài cho từng seat (trước hết bốc từ Bayesian range nếu không trùng lá)
                for &s in in_hand_seats {
                    let already_known = s != hero_seat
                        && opp_branch.state.players[s].hole_cards[0].is_some()
                        && opp_branch.state.players[s].hole_cards[1].is_some();
                    if already_known {
                        continue;
                    }
                    let assigned = sample_hand_from_bayesian_pool(
                        ranges.get(&s).map(|(h, w, _)| (h.as_slice(), w.as_slice())),
                        &mut used,
                        &mut rng,
                    );
                    if let Some([c0, c1]) = assigned {
                        opp_branch.state.players[s].hole_cards = [Some(c0), Some(c1)];
                    }
                }
                let rem_deck: Vec<Card> = (0..52)
                    .filter(|&idx| !used[idx])
                    .map(|idx| Card(idx as u8))
                    .collect();
                let mut shuffled = rem_deck;
                for i in (1..shuffled.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    shuffled.swap(i, j);
                }
                opp_branch.deck = Deck::from_remaining(&shuffled);

                let init_opp_stack = opp_branch.state.players[opp_target].stack;
                let opp_start_icm = start_icms.get(opp_target).copied().unwrap_or(0.0);
                self.traverse_multiway_cfr(
                    &mut opp_branch,
                    0,
                    iter_w,
                    opp_target,
                    init_opp_stack,
                    opp_start_icm,
                    0,
                    adv_net,
                    strat_net,
                    &mut infoset_table,
                    &mut rng,
                );
            }

            // Bước 2: Hero Traversal với bài thật `hero_hole` và bài đối thủ bốc từ Bayesian Range
            let mut branch = env.clone();
            let mut used = fixed_dead;
            for (i, p) in branch.state.players.iter().enumerate() {
                if i != hero_seat {
                    for c in p.hole_cards.iter().flatten() {
                        used[c.0 as usize] = true;
                    }
                }
            }
            for &s in &opp_seats {
                if branch.state.players[s].hole_cards[0].is_some()
                    && branch.state.players[s].hole_cards[1].is_some()
                {
                    continue;
                }
                if let Some([c0, c1]) = sample_hand_from_bayesian_pool(
                    ranges.get(&s).map(|(h, w, _)| (h.as_slice(), w.as_slice())),
                    &mut used,
                    &mut rng,
                ) {
                    branch.state.players[s].hole_cards = [Some(c0), Some(c1)];
                }
            }
            let mut rem_deck: Vec<Card> = (0..52)
                .filter(|&idx| !used[idx])
                .map(|idx| Card(idx as u8))
                .collect();
            for i in (1..rem_deck.len()).rev() {
                let j = rng.gen_range(0..=i);
                rem_deck.swap(i, j);
            }
            branch.deck = Deck::from_remaining(&rem_deck);

            let strat = DeepCFRPolicy::compute_regret_matching_probs(&root_regrets, &mask);
            let mut cur_vals = [0.0f32; NUM_ACTIONS];
            let mut expected_v = 0.0f32;

            for &a in &valid_actions {
                let a_idx = a.to_index();
                let mut child = branch.clone();
                let done = child.step(a).unwrap_or(true);
                let v = if done {
                    self.evaluate_multiway_outcome(
                        &child,
                        hero_seat,
                        hero_initial_stack,
                        start_icms.get(hero_seat).copied().unwrap_or(0.0),
                    )
                } else {
                    // Ghi nhận phản ứng đầu tiên của đối thủ nếu child đến lượt đối thủ
                    if child.state.current_player_idx != hero_seat {
                        let opp_cur = child.state.current_player_idx;
                        let opp_mask = child.get_action_mask();
                        if opp_mask.0 != 0 {
                            let opp_hole = match (
                                child.state.players[opp_cur].hole_cards[0],
                                child.state.players[opp_cur].hole_cards[1],
                            ) {
                                (Some(c0), Some(c1)) => [c0, c1],
                                _ => [Card(0), Card(1)],
                            };
                            let opp_in = child.state.players.iter().filter(|p| p.is_in_hand()).count();
                            let bucket = fast_hand_bucket(
                                opp_hole,
                                &child.state.community_cards,
                                opp_in.saturating_sub(1),
                            );
                            let opp_key = (opp_cur as u8, bucket, hash_seq(0, a_idx));
                            let opp_probs = if let Some(node) = infoset_table.get(&opp_key) {
                                node.average_strategy(&opp_mask)
                            } else {
                                blueprint_probs_for_state(&child.state, opp_cur, &opp_mask, adv_net, strat_net)
                            };
                            let sampled_opp = sample_action_from_probs(&opp_probs, &opp_mask, &mut rng);
                            opp_resp_total[a_idx] += 1;
                            match sampled_opp {
                                Action::Fold => opp_fold_cnt[a_idx] += 1,
                                Action::CheckCall => opp_call_cnt[a_idx] += 1,
                                _ => opp_raise_cnt[a_idx] += 1,
                            }
                        }
                    }

                    self.traverse_multiway_cfr(
                        &mut child,
                        1,
                        iter_w,
                        hero_seat,
                        hero_initial_stack,
                        start_icms.get(hero_seat).copied().unwrap_or(0.0),
                        hash_seq(0, a_idx),
                        adv_net,
                        strat_net,
                        &mut infoset_table,
                        &mut rng,
                    )
                };

                cur_vals[a_idx] = v;
                sum_v[a_idx] += v as f64;
                sum_sq_v[a_idx] += (v as f64) * (v as f64);
                expected_v += strat[a_idx] * v;
            }

            for &a in &valid_actions {
                let i = a.to_index();
                root_regrets[i] = (root_regrets[i] + (cur_vals[i] - expected_v)).max(0.0);
                root_policy_sum[i] += iter_w * strat[i];
            }
        }

        let mut action_evs = [f32::NEG_INFINITY; NUM_ACTIONS];
        let mut action_ev_se = [0.0f32; NUM_ACTIONS];
        let mut opp_fold_pct = [0.0f32; NUM_ACTIONS];
        let mut opp_call_pct = [0.0f32; NUM_ACTIONS];
        let mut opp_raise_pct = [0.0f32; NUM_ACTIONS];

        let n_f = iters as f64;
        for &a in &valid_actions {
            let i = a.to_index();
            let mean = sum_v[i] / n_f;
            let var = (sum_sq_v[i] / n_f - mean * mean).max(0.0);
            action_evs[i] = mean as f32;
            action_ev_se[i] = (var / n_f).sqrt() as f32;
            if opp_resp_total[i] > 0 {
                let t = opp_resp_total[i] as f32;
                opp_fold_pct[i] = opp_fold_cnt[i] as f32 / t;
                opp_call_pct[i] = opp_call_cnt[i] as f32 / t;
                opp_raise_pct[i] = opp_raise_cnt[i] as f32 / t;
            }
        }

        let mut action_probs = [0.0f32; NUM_ACTIONS];
        let total_p: f32 = valid_actions.iter().map(|a| root_policy_sum[a.to_index()]).sum();
        if total_p > 1e-4 {
            for &a in &valid_actions {
                let i = a.to_index();
                action_probs[i] = root_policy_sum[i] / total_p;
            }
        } else {
            action_probs = DeepCFRPolicy::compute_regret_matching_probs(&root_regrets, &mask);
        }

        let tie_eps = if self.config.use_icm { 1e-3 } else { 1e-2 };
        let mut best_action = valid_actions[0];
        let mut best_ev = action_evs[best_action.to_index()];
        let mut best_p = action_probs[best_action.to_index()];
        for &a in &valid_actions[1..] {
            let idx = a.to_index();
            let ev = action_evs[idx];
            let p = action_probs[idx];
            if ev > best_ev + tie_eps || ((ev - best_ev).abs() <= tie_eps && p > best_p) {
                best_ev = ev;
                best_p = p;
                best_action = a;
            }
        }

        let hero_is_ip = opp_seats
            .iter()
            .all(|&s| !env.state.players[s].is_active() || env.street_has_acted[s]);

        SubgameResult {
            best_action,
            action_probs,
            action_evs,
            trigger,
            iterations_run: iters,
            duration_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            diagnostics: SubgameDiagnostics {
                solver_kind: "N-Player Multiway CFR+",
                num_in_hand: in_hand_seats.len(),
                hero_is_ip,
                opp_range_size: opp_range_size_total,
                opp_bayesian_ess: if opp_ess_cnt > 0 {
                    opp_ess_sum / opp_ess_cnt as f32
                } else {
                    0.0
                },
                bayesian_history_steps: total_hist_steps,
                hero_showdown_eq,
                action_ev_se,
                opp_fold_pct,
                opp_call_pct,
                opp_raise_pct,
            },
        }
    }

    /// Giải Subgame tại thời điểm thực tế (Real-Time Resolving)
    pub fn resolve(
        &self,
        env: &Environment,
        hero_seat: usize,
        adv_net: &MLP,
        strat_net: Option<&MLP>,
        trigger: SearchTrigger,
        root_mask_override: Option<ActionMask>,
    ) -> SubgameResult {
        let start_time = Instant::now();
        let mask = root_mask_override.unwrap_or_else(|| env.get_action_mask());
        let valid_actions = mask.valid_actions();

        let in_hand_seats: Vec<usize> = env
            .state
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_in_hand())
            .map(|(i, _)| i)
            .collect();

        if valid_actions.len() <= 1 || in_hand_seats.len() <= 1 {
            let act = valid_actions.first().copied().unwrap_or(Action::CheckCall);
            let mut probs = [0.0f32; NUM_ACTIONS];
            probs[act.to_index()] = 1.0;
            return SubgameResult {
                best_action: act,
                action_probs: probs,
                action_evs: [0.0f32; NUM_ACTIONS],
                trigger,
                iterations_run: 0,
                duration_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                diagnostics: SubgameDiagnostics {
                    solver_kind: "Single-Action Trivial",
                    num_in_hand: in_hand_seats.len(),
                    ..Default::default()
                },
            };
        }

        if in_hand_seats.len() == 2 {
            let opp_seat = in_hand_seats
                .iter()
                .copied()
                .find(|&s| s != hero_seat)
                .unwrap_or(in_hand_seats[0]);
            return self.solve_heads_up_2sided_cfr(
                env,
                hero_seat,
                opp_seat,
                adv_net,
                strat_net,
                trigger,
                mask,
                start_time,
            );
        }

        self.solve_multiway_cfr(
            env,
            hero_seat,
            &in_hand_seats,
            adv_net,
            strat_net,
            trigger,
            mask,
            start_time,
        )
    }
}

fn sample_hand_from_bayesian_pool<R: Rng>(
    pool_opt: Option<(&[[Card; 2]], &[f32])>,
    used: &mut [bool; 52],
    rng: &mut R,
) -> Option<[Card; 2]> {
    if let Some((hands, weights)) = pool_opt {
        let mut candidates = Vec::new();
        let mut c_weights = Vec::new();
        for (&[c0, c1], &w) in hands.iter().zip(weights.iter()) {
            if !used[c0.0 as usize] && !used[c1.0 as usize] && w > 0.0 {
                candidates.push([c0, c1]);
                c_weights.push(w);
            }
        }
        if !candidates.is_empty() {
            let picked = if let Ok(dist) = WeightedIndex::new(&c_weights) {
                candidates[dist.sample(rng)]
            } else {
                candidates[rng.gen_range(0..candidates.len())]
            };
            used[picked[0].0 as usize] = true;
            used[picked[1].0 as usize] = true;
            return Some(picked);
        }
    }

    // Fallback an toàn khi pool đều va chạm bài: bốc 2 lá bất kỳ còn trống
    let avail: Vec<Card> = (0..52)
        .filter(|&idx| !used[idx])
        .map(|idx| Card(idx as u8))
        .collect();
    if avail.len() < 2 {
        return None;
    }
    let i0 = rng.gen_range(0..avail.len());
    let mut i1 = rng.gen_range(0..avail.len() - 1);
    if i1 >= i0 {
        i1 += 1;
    }
    let c0 = avail[i0];
    let c1 = avail[i1];
    used[c0.0 as usize] = true;
    used[c1.0 as usize] = true;
    Some([c0, c1])
}

fn sample_action_from_probs<R: Rng>(probs: &[f32; NUM_ACTIONS], mask: &ActionMask, rng: &mut R) -> Action {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::state::ActionRecord;
    use crate::poker_core::{Card, Rank, Suit};

    #[test]
    fn test_subgame_solver_nuts_facing_check() {
        let n = 2;
        let stacks = vec![2000, 2000];
        let mut env = Environment::new(n, &stacks, 20, 10, 0);

        env.state.community_cards = vec![
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Queen, Suit::Hearts),
            Card::new(Rank::Jack, Suit::Hearts),
            Card::new(Rank::Two, Suit::Clubs),
        ];
        env.state.street = Street::River;
        env.state.pot = 400;
        env.state.current_player_idx = 0;

        env.state.players[0].hole_cards = [
            Some(Card::new(Rank::Ten, Suit::Hearts)),
            Some(Card::new(Rank::Nine, Suit::Hearts)),
        ];
        env.state.players[1].hole_cards = [
            Some(Card::new(Rank::Two, Suit::Diamonds)),
            Some(Card::new(Rank::Three, Suit::Diamonds)),
        ];

        let adv_mlp = MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]);
        let solver = SubgameSolver::with_default_config();

        let trigger = SearchTrigger::HighStakesPot { pot_bb: 20.0 };
        let result = solver.resolve(&env, 0, &adv_mlp, None, trigger, None);

        let bet_ev = result.action_evs[Action::BetPot75.to_index()]
            .max(result.action_evs[Action::BetPot33.to_index()])
            .max(result.action_evs[Action::AllIn.to_index()]);
        let fold_ev = result.action_evs[Action::Fold.to_index()];
        assert!(bet_ev > fold_ev, "Cầm bài Nuts thì Bet/Raise EV phải cao hơn Fold!");
        #[cfg(not(debug_assertions))]
        assert!(result.duration_ms < 100.0, "Thời gian giải phải dưới 100ms trong Release mode!");
        #[cfg(debug_assertions)]
        assert!(result.duration_ms < 5000.0, "Thời gian giải phải dưới 5000ms trong Debug mode!");
    }

    #[test]
    fn test_subgame_solver_pure_air_facing_allin() {
        let n = 2;
        let stacks = vec![1000, 1000];
        let mut env = Environment::new(n, &stacks, 20, 10, 0);

        env.state.community_cards = vec![
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::King, Suit::Diamonds),
            Card::new(Rank::Queen, Suit::Spades),
            Card::new(Rank::Four, Suit::Clubs),
            Card::new(Rank::Five, Suit::Diamonds),
        ];
        env.state.street = Street::River;
        env.state.pot = 1000;
        env.state.current_highest_bet = 1000;
        env.state.players[1].stack = 0;
        env.state.players[1].is_all_in = true;
        env.state.players[1].current_bet = 1000;
        env.state.players[1].total_contributed = 1000;
        env.state.players[1].hole_cards = [
            Some(Card::new(Rank::Ace, Suit::Spades)),
            Some(Card::new(Rank::King, Suit::Clubs)),
        ];
        env.state.current_player_idx = 0;

        env.state.players[0].hole_cards = [
            Some(Card::new(Rank::Seven, Suit::Clubs)),
            Some(Card::new(Rank::Two, Suit::Diamonds)),
        ];

        let adv_mlp = MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]);
        let solver = SubgameSolver::with_default_config();

        let trigger = SearchTrigger::UnusualOverbet { bet_pot_ratio: 2.0 };
        let result = solver.resolve(&env, 0, &adv_mlp, None, trigger, None);

        let fold_ev = result.action_evs[Action::Fold.to_index()];
        let call_ev = result.action_evs[Action::CheckCall.to_index()];
        assert!(fold_ev >= call_ev, "Cầm bài rác 72o đối mặt Shove thì Fold EV phải cao hơn Call EV!");
    }

    #[test]
    fn test_subgame_solver_third_pair_river_sampling_blocks_suicidal_allin() {
        let n = 2;
        let stacks = vec![1448, 2732];
        let mut env = Environment::new(n, &stacks, 40, 20, 0);

        env.state.community_cards = vec![
            Card::new(Rank::Three, Suit::Hearts),
            Card::new(Rank::Five, Suit::Hearts),
            Card::new(Rank::Six, Suit::Clubs),
            Card::new(Rank::Eight, Suit::Diamonds),
            Card::new(Rank::King, Suit::Hearts),
        ];
        env.state.street = Street::River;
        env.state.pot = 360;
        env.state.players[0].total_contributed = 180;
        env.state.players[1].total_contributed = 180;
        env.state.current_highest_bet = 0;
        env.state.current_player_idx = 0;

        env.state.players[0].hole_cards = [
            Some(Card::new(Rank::Four, Suit::Clubs)),
            Some(Card::new(Rank::Five, Suit::Spades)),
        ];
        env.state.players[1].hole_cards = [None, None];

        let (adv_mlp, strat_mlp) = if let (Ok(adv), Ok(strat)) = (
            MLP::load_from_file("deep_cfr_model.json"),
            MLP::load_from_file("deep_cfr_strategy.json"),
        ) {
            if adv.input_dim() == crate::environment::FEATURE_DIM {
                (adv, Some(strat))
            } else {
                (MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]), None)
            }
        } else {
            (MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]), None)
        };
        let mut config = SubgameSolverConfig::default();
        config.iterations = 120;
        let solver = SubgameSolver::new(config);

        let trigger = SearchTrigger::ManualForced;
        let result = solver.resolve(&env, 0, &adv_mlp, strat_mlp.as_ref(), trigger, None);

        let non_allin_ev = result.action_evs[Action::BetPot33.to_index()]
            .max(result.action_evs[Action::CheckCall.to_index()])
            .max(result.action_evs[Action::BetSmall.to_index()]);
        let allin_ev = result.action_evs[Action::AllIn.to_index()];
        assert!(
            non_allin_ev >= allin_ev,
            "Cầm đôi 5 trên river board K+flush, action kiểm soát EV ({:.2}) phải >= AllIn EV ({:.2})",
            non_allin_ev,
            allin_ev
        );
        assert_ne!(result.best_action, Action::AllIn, "Solver không được phép All-In tự sát với đôi 5!");
    }

    #[test]
    fn test_subgame_solver_preflop_trash_sampling_blocks_aggressive_reraise() {
        let n = 4;
        let stacks = vec![750, 2498, 2852, 890];
        let mut env = Environment::new(n, &stacks, 40, 20, 0);

        env.state.street = Street::Preflop;
        env.state.pot = 100;
        env.state.current_highest_bet = 40;
        env.state.current_player_idx = 1;
        env.state.players[0].current_bet = 40;
        env.state.players[2].current_bet = 20;
        env.state.players[3].current_bet = 40;

        env.state.players[1].hole_cards = [
            Some(Card::new(Rank::Ten, Suit::Clubs)),
            Some(Card::new(Rank::Two, Suit::Spades)),
        ];
        env.state.players[0].hole_cards = [None, None];
        env.state.players[2].hole_cards = [None, None];
        env.state.players[3].hole_cards = [None, None];

        let (adv_mlp, strat_mlp) = if let (Ok(adv), Ok(strat)) = (
            MLP::load_from_file("deep_cfr_model.json"),
            MLP::load_from_file("deep_cfr_strategy.json"),
        ) {
            if adv.input_dim() == crate::environment::FEATURE_DIM {
                (adv, Some(strat))
            } else {
                (MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]), None)
            }
        } else {
            (MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]), None)
        };
        let mut config = SubgameSolverConfig::default();
        config.iterations = 120;
        let solver = SubgameSolver::new(config);

        let trigger = SearchTrigger::ManualForced;
        let result = solver.resolve(&env, 1, &adv_mlp, strat_mlp.as_ref(), trigger, None);

        let fold_ev = result.action_evs[Action::Fold.to_index()];
        let bet75_ev = result.action_evs[Action::BetPot75.to_index()];
        assert!(
            fold_ev >= bet75_ev || result.best_action != Action::BetPot75,
            "T2o không được phép coi BetPot75 là tối ưu áp đảo!"
        );
        assert_ne!(result.best_action, Action::AllIn, "T2o không được All-In!");
    }

    #[test]
    fn test_subgame_solver_heads_up_duel_auto_trigger() {
        let n = 2;
        let stacks = vec![3000, 1000];
        let mut env = Environment::new(n, &stacks, 40, 20, 0);
        env.state.street = Street::Preflop;
        env.state.pot = 60;
        env.state.current_player_idx = 0;

        let solver = SubgameSolver::with_default_config();
        let trigger = solver.should_trigger_search(&env, 0, false);
        assert_eq!(trigger, SearchTrigger::HeadsUpDuel);
    }

    #[test]
    fn test_subgame_solver_multiway_folded_to_two_does_not_trigger_heads_up_duel() {
        let n = 8;
        let stacks = vec![1000; 8];
        let mut env = Environment::new(n, &stacks, 40, 20, 0);
        env.state.street = Street::Preflop;
        env.state.pot = 60;
        for i in 2..8 {
            env.state.players[i].is_folded = true;
        }
        env.state.current_player_idx = 0;

        let solver = SubgameSolver::with_default_config();
        let trigger = solver.should_trigger_search(&env, 0, false);
        assert_eq!(trigger, SearchTrigger::FastPolicyFallback);
    }

    #[test]
    fn test_subgame_solver_overbet_trigger_multiway() {
        let stacks = vec![2000; 6];
        let mut env = Environment::new(6, &stacks, 40, 20, 0);
        env.state.is_cash_game = true;
        env.state.street = Street::Flop;
        env.state.community_cards = vec![
            Card::from_str_repr("As").unwrap(),
            Card::from_str_repr("8d").unwrap(),
            Card::from_str_repr("2c").unwrap(),
        ];
        env.state.pot = 250;
        env.state.current_highest_bet = 150;
        env.state.current_player_idx = 0;

        let solver = SubgameSolver::with_default_config();
        let trigger = solver.should_trigger_search(&env, 0, false);
        match trigger {
            SearchTrigger::UnusualOverbet { bet_pot_ratio } => {
                assert!(
                    (bet_pot_ratio - 1.5).abs() < 1e-4,
                    "Expected bet_pot_ratio 1.5, got {}",
                    bet_pot_ratio
                );
            }
            other => panic!("Expected UnusualOverbet, got {:?}", other),
        }
    }

    #[test]
    fn test_subgame_solver_10max_cash_no_deck_exhaustion_and_chip_ev() {
        let mut env = Environment::new(10, &[10_000; 10], 100, 50, 0);
        env.state.is_cash_game = true;
        env.state.street = Street::Flop;
        env.state.community_cards = vec![
            Card::from_str_repr("Ad").unwrap(),
            Card::from_str_repr("Ac").unwrap(),
            Card::from_str_repr("2s").unwrap(),
        ];
        env.state.pot = 3000;
        env.state.current_highest_bet = 300;
        env.state.current_player_idx = 0;

        for p in &mut env.state.players {
            p.stack = 9700;
            p.current_bet = 300;
            p.total_contributed = 300;
            p.hole_cards = [None, None];
        }
        env.state.players[0].hole_cards = [
            Some(Card::from_str_repr("As").unwrap()),
            Some(Card::from_str_repr("Ah").unwrap()),
        ];

        let adv_mlp = MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]);
        let mut cfg = SubgameSolverConfig::lounge_cash_default();
        cfg.iterations = 40;
        let solver = SubgameSolver::new(cfg);

        let trigger = solver.should_trigger_search(&env, 0, false);
        assert_ne!(trigger, SearchTrigger::FastPolicyFallback);

        let res = solver.resolve(&env, 0, &adv_mlp, None, trigger, None);
        assert_ne!(res.best_action, Action::Fold, "Quad Aces must never fold in 10-Max Cash");
        for a in env.get_action_mask().valid_actions() {
            assert!(res.action_evs[a.to_index()].is_finite());
        }
    }

    #[test]
    fn test_bayesian_range_weighting_uniform_and_history() {
        let mut rng = thread_rng();
        let mut env = Environment::new(2, &[500, 500], 20, 10, 0);
        env.state.is_cash_game = true;
        env.state.street = Street::Flop;
        env.state.community_cards = vec![
            Card::from_str_repr("Ah").unwrap(),
            Card::from_str_repr("Kd").unwrap(),
            Card::from_str_repr("2c").unwrap(),
        ];
        env.state.pot = 120;
        env.state.players[0].hole_cards = [
            Some(Card::from_str_repr("Qc").unwrap()),
            Some(Card::from_str_repr("Jc").unwrap()),
        ];
        env.state.players[1].hole_cards = [None, None];

        let adv_mlp = MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]);

        // 1. Không có action_history -> trọng số phải đều hoàn toàn (ESS == hands.len())
        let (h0, w0, steps0) = build_bayesian_range(&env, 1, None, None, 20, &adv_mlp, None, &mut rng);
        assert_eq!(h0.len(), 20);
        assert_eq!(steps0, 0);
        assert!((w0.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        let ess0 = kish_ess(&w0);
        assert!(
            (ess0 - 20.0).abs() < 1e-2,
            "Without history, prior must be uniform with ESS == 20, got {}",
            ess0
        );

        // 2. Có action_history -> trọng số chuẩn hóa Bayesian hợp lệ và ESS <= N
        env.state.players[1].total_contributed = 60;
        env.state.players[1].stack = 440;
        env.state.action_history = vec![
            ActionRecord {
                player_idx: 1,
                street: Street::Preflop,
                action: Action::BetPot75,
                amount: 40,
            },
            ActionRecord {
                player_idx: 1,
                street: Street::Flop,
                action: Action::BetPot33,
                amount: 20,
            },
        ];
        let (h1, w1, steps1) = build_bayesian_range(&env, 1, None, None, 20, &adv_mlp, None, &mut rng);
        assert_eq!(h1.len(), 20);
        assert_eq!(steps1, 2);
        assert!((w1.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(w1.iter().all(|&w| w.is_finite() && w > 0.0));
        assert!(kish_ess(&w1) <= 20.01);
    }

    #[test]
    fn test_2sided_cfr_k6_river_lounge_spot_never_suicidal_allin() {
        // Tình huống thực chiến của user: [Kd 6d] trên [6s Ah Tc 3d 7s], River, 2 người in hand / 7 ghế, Pot 125, Stack 166, BB 2
        let (adv_mlp, strat_mlp) = if let (Ok(adv), Ok(strat)) = (
            MLP::load_from_file("deep_cfr_lounge_model.json"),
            MLP::load_from_file("deep_cfr_lounge_strategy.json"),
        ) {
            if adv.input_dim() == crate::environment::FEATURE_DIM {
                (adv, Some(strat))
            } else {
                (MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]), None)
            }
        } else {
            (MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]), None)
        };

        let mut cfg = SubgameSolverConfig::lounge_cash_default();
        cfg.iterations = 160;
        let solver = SubgameSolver::new(cfg);

        for hero_is_ip in [true, false] {
            let mut stacks = vec![200u32; 7];
            stacks[0] = 166;
            stacks[1] = 166;
            let mut env = Environment::new(7, &stacks, 2, 1, 0);
            env.state.is_cash_game = true;
            env.state.street = Street::River;
            env.state.pot = 125;
            env.state.current_highest_bet = 0;
            env.state.current_player_idx = 0;
            env.state.community_cards = vec![
                Card::from_str_repr("6s").unwrap(),
                Card::from_str_repr("Ah").unwrap(),
                Card::from_str_repr("Tc").unwrap(),
                Card::from_str_repr("3d").unwrap(),
                Card::from_str_repr("7s").unwrap(),
            ];
            for i in 2..7 {
                env.state.players[i].is_folded = true;
            }
            env.state.players[0].total_contributed = 63;
            env.state.players[1].total_contributed = 62;
            env.state.players[0].hole_cards = [
                Some(Card::from_str_repr("Kd").unwrap()),
                Some(Card::from_str_repr("6d").unwrap()),
            ];
            env.state.players[1].hole_cards = [None, None];
            env.street_has_acted = vec![false; 7];
            if hero_is_ip {
                env.street_has_acted[1] = true;
            }

            let res = solver.resolve(
                &env,
                0,
                &adv_mlp,
                strat_mlp.as_ref(),
                SearchTrigger::ManualForced,
                None,
            );
            println!(
                "\n[K6 River Lounge IP={}] kind={} ESS={:.1} Eq={:.1}% Best={:?}",
                hero_is_ip,
                res.diagnostics.solver_kind,
                res.diagnostics.opp_bayesian_ess,
                res.diagnostics.hero_showdown_eq * 100.0,
                res.best_action
            );
            for a in Action::ALL {
                let i = a.to_index();
                println!(
                    "  {:<15}: EV {:>+6.2} (±{:.2}) | GTO {:>5.1}% | Opp F:{:>4.1}% C:{:>4.1}% R:{:>4.1}%",
                    a.name(),
                    res.action_evs[i],
                    res.diagnostics.action_ev_se[i],
                    res.action_probs[i] * 100.0,
                    res.diagnostics.opp_fold_pct[i] * 100.0,
                    res.diagnostics.opp_call_pct[i] * 100.0,
                    res.diagnostics.opp_raise_pct[i] * 100.0
                );
            }

            assert_ne!(
                res.best_action,
                Action::AllIn,
                "2-sided Nash CFR+ must never shove 83BB with K6o (3rd pair) on A-T-7-6-3 when IP={}",
                hero_is_ip
            );
            assert!(
                res.diagnostics.action_ev_se[Action::AllIn.to_index()].is_finite(),
                "SE must be finite"
            );
            assert!(
                res.diagnostics.opp_call_pct[Action::AllIn.to_index()] > 0.15,
                "Nash opponent must defend/call River All-In with >15% of range, got {:.1}%",
                res.diagnostics.opp_call_pct[Action::AllIn.to_index()] * 100.0
            );
        }
    }

    #[test]
    fn test_2sided_cfr_nuts_gets_called_not_clairvoyant() {
        let mut stacks = vec![400u32; 2];
        stacks[0] = 400;
        stacks[1] = 400;
        let mut env = Environment::new(2, &stacks, 4, 2, 0);
        env.state.is_cash_game = true;
        env.state.street = Street::River;
        env.state.pot = 200;
        env.state.current_player_idx = 0;
        env.state.community_cards = vec![
            Card::from_str_repr("As").unwrap(),
            Card::from_str_repr("Ks").unwrap(),
            Card::from_str_repr("Qs").unwrap(),
            Card::from_str_repr("2d").unwrap(),
            Card::from_str_repr("7c").unwrap(),
        ];
        env.state.players[0].hole_cards = [
            Some(Card::from_str_repr("Js").unwrap()),
            Some(Card::from_str_repr("Ts").unwrap()),
        ];
        env.state.players[1].hole_cards = [None, None];
        env.street_has_acted = vec![false, true]; // Hero IP

        let adv_mlp = MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]);
        let mut cfg = SubgameSolverConfig::lounge_cash_default();
        cfg.iterations = 120;
        let solver = SubgameSolver::new(cfg);

        let res = solver.resolve(&env, 0, &adv_mlp, None, SearchTrigger::ManualForced, None);
        assert!(res.action_evs[Action::BetPot75.to_index()] > res.action_evs[Action::CheckCall.to_index()]);
        assert!(
            res.diagnostics.opp_call_pct[Action::BetPot33.to_index()] > 0.0,
            "Non-clairvoyant opponent must call Hero's value bets at positive frequency"
        );
    }
}

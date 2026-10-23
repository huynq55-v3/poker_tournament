//! Subgame Solver (Depth-Limited Solving / Lookahead Search) cho Poker AI
//!
//! Dựa trên nguyên lý Continual Resolving & Subgame Solving của DeepStack, Libratus và ReBeL:
//! Khi đến lượt Hero ra quyết định trong ván đấu thực tế:
//! 1. Kiểm tra bộ lọc kích hoạt (Hybrid Trigger): Pot lớn, đối thủ overbet dị, hoặc các vòng cược nhạy cảm (Turn/River).
//! 2. Dựng cây trò chơi con (Mini-subgame Tree) từ trạng thái hiện tại tới độ sâu giới hạn (max_depth).
//! 3. Định giá các nút lá (Leaf Nodes) bằng Showdown Pot-Equity kết hợp Advantage Network.
//! 4. Chạy nhanh 50 - 150 vòng Monte Carlo CFR (MCCFR) thời gian thực trên cây con.
//! 5. Trích xuất Action EV chính xác (tính bằng BB) và Chiến thuật cân bằng Nash cục bộ.

use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::equity::equity_vs_random;
use crate::equity::icm::{calculate_icm_ev, get_sng_payouts};
use crate::neural_network::MLP;
use crate::deep_cfr::policy::DeepCFRPolicy;
use crate::poker_core::card::Card;
use crate::poker_core::deck::Deck;
use crate::poker_core::evaluator::{evaluate_7_cards, HandCategory, HandRank};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use std::time::Instant;

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
            SearchTrigger::HeadsUpDuel => write!(f, "⚔️ Chung kết 1-vs-1 Heads-Up (Chỉ còn 2 người trong tour)"),
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
    pub max_depth: usize,             // Độ sâu tối đa số actions (mặc định 3)
    pub iterations: usize,            // Số vòng lặp CFR cục bộ (mặc định 60)
    pub min_pot_bb_trigger: f32,      // Kích hoạt khi pot >= X BB (mặc định 12.0)
    pub unusual_bet_ratio_trigger: f32, // Kích hoạt khi cược > 1.10 pot (Overbet)
    pub search_turn_river: bool,      // Tự động tìm kiếm ở Turn và River
    pub use_icm: bool,                // Đánh giá EV theo Tournament ICM (mặc định true)
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

/// Kết quả tính toán từ Subgame Solver
#[derive(Clone, Debug)]
pub struct SubgameResult {
    pub best_action: Action,
    pub action_probs: [f32; NUM_ACTIONS],
    pub action_evs: [f32; NUM_ACTIONS], // EV kỳ vọng (Tournament ICM Delta * 10 hoặc Chip-EV)
    pub trigger: SearchTrigger,
    pub iterations_run: usize,
    pub duration_ms: f64,
}

/// Đánh giá độ hợp lý thực tế của tay bài đối thủ kết hợp CẢ BÀI TẨY VÀ BÀI CHUNG (Postflop Made Hand & Board Texture):
/// Khắc phục triệt để nhược điểm của việc chỉ lọc theo Chen Score thuần:
/// 1. "Có khi đối thủ cầm bài rác preflop mà trúng bài chung thì vẫn win!":
///    - Nếu bài rác trúng bài chung (One Pair, Two Pair, Trips, Straight, Flush): HOÀN TOÀN HỢP LỆ VÀ GIỮ LẠI 100%!
/// 2. Có Draw (Flush Draw 4 lá cùng chất trên mặt bài, Straight Draw): Giữ lại 100%.
/// 3. Cầm Overcards / Bài tẩy mạnh preflop (Chen >= 0.25 như AK, AQ, AJ dù chưa trúng board): Giữ lại 100%.
/// 4. Pure Air hoàn toàn không trúng gì: Vẫn giữ lại ~25% số lần để mô phỏng tần suất Bluff / C-bet tự nhiên của đối thủ!
pub fn is_plausible_opponent_hand<R: Rng>(
    c0: Card,
    c1: Card,
    community_cards: &[Card],
    rng: &mut R,
) -> bool {
    if community_cards.len() >= 3 {
        // 1. Tự cầm Pocket Pair (22, 33, ... AA): luôn hợp lệ!
        if c0.rank_val() == c1.rank_val() {
            return true;
        }

        // 2. Dù là bài rác preflop (như 83o, 74o, 62o), nếu TRÚNG BÀI CHUNG (One Pair, Two Pair, Trips):
        //    -> "Có khi đối thủ cầm bài rác mà win đấy!" -> HOÀN TOÀN HỢP LỆ, KHÔNG ĐƯỢC LOẠI BỎ!
        let hits_board = community_cards
            .iter()
            .any(|b| b.rank_val() == c0.rank_val() || b.rank_val() == c1.rank_val());
        if hits_board {
            return true;
        }

        // 3. Đánh giá Made Hand đầy đủ (Straight, Flush, Full House, ...) bằng evaluate_7_cards
        let mut cards = Vec::with_capacity(community_cards.len() + 2);
        cards.extend_from_slice(community_cards);
        cards.push(c0);
        cards.push(c1);
        let rank = evaluate_7_cards(&cards);
        if rank.category() >= HandCategory::OnePair {
            return true;
        }

        // 4. Kiểm tra Flush Draw (4 lá cùng chất trên mặt bài)
        let s0 = c0.suit_val();
        let s1 = c1.suit_val();
        let s0_count = 1 + community_cards.iter().filter(|b| b.suit_val() == s0).count();
        let s1_count = 1 + community_cards.iter().filter(|b| b.suit_val() == s1).count();
        if s0_count >= 4 || s1_count >= 4 {
            return true;
        }

        // 5. Cầm bài tẩy chất lượng cao chưa trúng board (Overcards như AK, AQ, AJ - Chen >= 0.25):
        if crate::environment::state::chen_score(c0, c1) >= 0.25 {
            return true;
        }

        // 6. Nếu là bài hoàn toàn không trúng (Pure Air):
        //    Vẫn cho phép xuất hiện ~25% số lần để mô phỏng tần suất Bluff / C-bet tự nhiên của đối thủ!
        rng.gen_bool(0.25)
    } else {
        // Ở Preflop (chưa có bài chung):
        // Bài có tiềm năng (Chen >= 0.20) luôn hợp lệ.
        // Bài rác (Chen < 0.20) vẫn cho phép xuất hiện ~20% (mô phỏng Steal blind / Limp-call / Bluff).
        if crate::environment::state::chen_score(c0, c1) >= 0.20 {
            true
        } else {
            rng.gen_bool(0.20)
        }
    }
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

        // 0. TRUE HEADS-UP TOURNAMENT DUEL TRIGGER:
        // CHỈ tự động giải Solver cho mọi vòng cược khi giải đấu THỰC SỰ chỉ còn <= 2 người sống sót (True Heads-Up)!
        // Nếu bàn vẫn còn đông người (num_alive >= 3) nhưng 6 người fold:
        // ĐÂY LÀ VÁN BÀI MULTIWAY TOURNAMENT -> Phải tuân thủ áp lực ICM Bubble & Top 4 của Blueprint Net!
        let num_alive = env.state.players.iter().filter(|p| p.stack > 0 || p.is_all_in).count();
        if num_alive <= 2 {
            return SearchTrigger::HeadsUpDuel;
        }

        let bb = env.state.bb_size.max(1) as f32;
        let pot_bb = env.state.pot as f32 / bb;

        // 1. Pot lớn: Đòi hỏi tính toán chính xác
        if pot_bb >= self.config.min_pot_bb_trigger {
            return SearchTrigger::HighStakesPot { pot_bb };
        }

        // 2. Đối thủ overbet dị hoặc sizing bất thường
        let hero_bet = env.state.players[hero_seat].current_bet;
        let to_call = env.state.current_highest_bet.saturating_sub(hero_bet) as f32;
        let current_pot = env.state.pot as f32;
        if current_pot > 0.0 && (to_call / current_pot) >= self.config.unusual_bet_ratio_trigger {
            return SearchTrigger::UnusualOverbet {
                bet_pot_ratio: to_call / current_pot,
            };
        }

        // 3. Cú cược đe dọa stack của Hero (> 35% stack)
        let hero_stack = env.state.players[hero_seat].stack as f32;
        if hero_stack > 0.0 && (to_call / hero_stack) >= 0.35 {
            return SearchTrigger::DeepStackCommitment {
                to_call_ratio: to_call / hero_stack,
            };
        }

        // 4. Các vòng cược quyết định Postflop (Turn & River)
        if self.config.search_turn_river
            && (env.state.street == Street::Turn || env.state.street == Street::River)
        {
            return SearchTrigger::CrucialPostflopStreet {
                street: env.state.street,
            };
        }

        SearchTrigger::FastPolicyFallback
    }

    /// Tính toán payoff tại các nút kết thúc ván bài (Showdown hoặc đối thủ đã Fold hết)
    #[inline]
    fn compute_terminal_payoff(
        &self,
        env: &Environment,
        hero_seat: usize,
        initial_hero_stack: u32,
        hero_start_icm: f32,
    ) -> f32 {
        if self.config.use_icm {
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
                } else if env.state.community_cards.len() == 5 {
                    // Showdown thực tế: Đọ bài tìm người thắng cuộc
                    let mut best_score: Option<HandRank> = None;
                    let mut winners = Vec::new();
                    for &idx in &remaining_in_hand {
                        let p = &env.state.players[idx];
                        if let (Some(c0), Some(c1)) = (p.hole_cards[0], p.hole_cards[1]) {
                            let mut cards = env.state.community_cards.clone();
                            cards.push(c0);
                            cards.push(c1);
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
                    }
                    if !winners.is_empty() {
                        let win_share = env.state.pot as f32 / winners.len() as f32;
                        for &w in &winners {
                            end_stacks[w] += win_share;
                        }
                    } else {
                        let share = env.state.pot as f32 / remaining_in_hand.len() as f32;
                        for &idx in &remaining_in_hand {
                            end_stacks[idx] += share;
                        }
                    }
                } else if !remaining_in_hand.is_empty() {
                    let share = env.state.pot as f32 / remaining_in_hand.len() as f32;
                    for &idx in &remaining_in_hand {
                        end_stacks[idx] += share;
                    }
                }
            }
            let payouts = get_sng_payouts(end_stacks.len());
            let end_evs = calculate_icm_ev(&end_stacks, payouts);
            let hero_end_icm = end_evs.get(hero_seat).copied().unwrap_or(0.0);
            (hero_end_icm - hero_start_icm) * 10.0
        } else {
            let bb = env.state.bb_size.max(1) as f32;
            let hero_player = &env.state.players[hero_seat];
            let in_hand_count = env.state.players.iter().filter(|p| p.is_in_hand()).count();
            let final_stack = if !hero_player.is_folded {
                if in_hand_count <= 1 {
                    (hero_player.stack + env.state.pot) as f32
                } else if env.state.community_cards.len() == 5 {
                    let mut best_score: Option<HandRank> = None;
                    let mut winners = Vec::new();
                    for (i, p) in env.state.players.iter().enumerate() {
                        if p.is_in_hand() {
                            if let (Some(c0), Some(c1)) = (p.hole_cards[0], p.hole_cards[1]) {
                                let mut cards = env.state.community_cards.clone();
                                cards.push(c0);
                                cards.push(c1);
                                let rank = evaluate_7_cards(&cards);
                                match best_score {
                                    None => {
                                        best_score = Some(rank);
                                        winners.push(i);
                                    }
                                    Some(br) if rank > br => {
                                        best_score = Some(rank);
                                        winners.clear();
                                        winners.push(i);
                                    }
                                    Some(br) if rank == br => {
                                        winners.push(i);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    let hero_win_share = if winners.contains(&hero_seat) {
                        env.state.pot as f32 / winners.len() as f32
                    } else {
                        0.0
                    };
                    hero_player.stack as f32 + hero_win_share
                } else {
                    hero_player.stack as f32
                }
            } else {
                hero_player.stack as f32
            };
            (final_stack - initial_hero_stack as f32) / bb
        }
    }

    /// Định giá nút lá (Leaf Evaluation)
    /// Khi cây con đạt độ sâu giới hạn mà chưa kết thúc ván:
    /// Dùng Pot-Equity Share kết hợp Tournament ICM và Tactical Advantage Net
    #[inline]
    fn evaluate_leaf(
        &self,
        env: &Environment,
        hero_seat: usize,
        initial_hero_stack: u32,
        hero_start_icm: f32,
        adv_net: &MLP,
    ) -> f32 {
        let hero_player = &env.state.players[hero_seat];
        let in_hand_count = env
            .state
            .players
            .iter()
            .filter(|p| !p.is_folded && (p.stack > 0 || p.current_bet > 0 || p.is_all_in))
            .count();

        let base_ev = if self.config.use_icm {
            let mut end_stacks: Vec<f32> = env.state.players.iter().map(|p| p.stack as f32).collect();
            let pot = env.state.pot as f32;

            if hero_player.is_folded {
                let opp_count = in_hand_count;
                if opp_count > 0 {
                    let share = pot / opp_count as f32;
                    for (i, p) in env.state.players.iter().enumerate() {
                        if p.is_in_hand() {
                            end_stacks[i] += share;
                        }
                    }
                }
            } else if in_hand_count <= 1 {
                end_stacks[hero_seat] += pot;
            } else if env.state.community_cards.len() == 5 {
                let mut best_score: Option<HandRank> = None;
                let mut winners = Vec::new();
                for (i, p) in env.state.players.iter().enumerate() {
                    if p.is_in_hand() {
                        if let (Some(c0), Some(c1)) = (p.hole_cards[0], p.hole_cards[1]) {
                            let mut cards = env.state.community_cards.clone();
                            cards.push(c0);
                            cards.push(c1);
                            let rank = evaluate_7_cards(&cards);
                            match best_score {
                                None => {
                                    best_score = Some(rank);
                                    winners.push(i);
                                }
                                Some(br) if rank > br => {
                                    best_score = Some(rank);
                                    winners.clear();
                                    winners.push(i);
                                }
                                Some(br) if rank == br => {
                                    winners.push(i);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                if !winners.is_empty() {
                    let win_share = pot / winners.len() as f32;
                    for &w in &winners {
                        end_stacks[w] += win_share;
                    }
                } else {
                    let share = pot / in_hand_count.max(1) as f32;
                    for (i, p) in env.state.players.iter().enumerate() {
                        if p.is_in_hand() {
                            end_stacks[i] += share;
                        }
                    }
                }
            } else {
                let eq = if let (Some(c0), Some(c1)) = (hero_player.hole_cards[0], hero_player.hole_cards[1]) {
                    equity_vs_random([c0, c1], &env.state.community_cards, in_hand_count.saturating_sub(1))
                } else {
                    1.0 / in_hand_count as f32
                };
                end_stacks[hero_seat] += eq * pot;
                let opp_count = in_hand_count.saturating_sub(1);
                if opp_count > 0 {
                    let opp_share = ((1.0 - eq) * pot) / opp_count as f32;
                    for (i, p) in env.state.players.iter().enumerate() {
                        if i != hero_seat && p.is_in_hand() {
                            end_stacks[i] += opp_share;
                        }
                    }
                }
            }

            let payouts = get_sng_payouts(end_stacks.len());
            let end_evs = calculate_icm_ev(&end_stacks, payouts);
            let hero_end_icm = end_evs.get(hero_seat).copied().unwrap_or(0.0);
            (hero_end_icm - hero_start_icm) * 10.0
        } else {
            let bb = env.state.bb_size.max(1) as f32;
            if hero_player.is_folded {
                (hero_player.stack as f32 - initial_hero_stack as f32) / bb
            } else if in_hand_count <= 1 {
                ((hero_player.stack + env.state.pot) as f32 - initial_hero_stack as f32) / bb
            } else if env.state.community_cards.len() == 5 {
                let mut best_score: Option<HandRank> = None;
                let mut winners = Vec::new();
                for (i, p) in env.state.players.iter().enumerate() {
                    if p.is_in_hand() {
                        if let (Some(c0), Some(c1)) = (p.hole_cards[0], p.hole_cards[1]) {
                            let mut cards = env.state.community_cards.clone();
                            cards.push(c0);
                            cards.push(c1);
                            let rank = evaluate_7_cards(&cards);
                            match best_score {
                                None => {
                                    best_score = Some(rank);
                                    winners.push(i);
                                }
                                Some(br) if rank > br => {
                                    best_score = Some(rank);
                                    winners.clear();
                                    winners.push(i);
                                }
                                Some(br) if rank == br => {
                                    winners.push(i);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                let hero_win_share = if winners.contains(&hero_seat) {
                    env.state.pot as f32 / winners.len() as f32
                } else {
                    0.0
                };
                let expected_chips = hero_player.stack as f32 + hero_win_share;
                (expected_chips - initial_hero_stack as f32) / bb
            } else {
                let eq = if let (Some(c0), Some(c1)) = (hero_player.hole_cards[0], hero_player.hole_cards[1]) {
                    equity_vs_random([c0, c1], &env.state.community_cards, in_hand_count.saturating_sub(1))
                } else {
                    1.0 / in_hand_count as f32
                };
                let expected_chips = hero_player.stack as f32 + (eq * env.state.pot as f32);
                (expected_chips - initial_hero_stack as f32) / bb
            }
        };

        // Bổ sung Tactical Bonus từ Advantage Net (mạng nơ-ron cũng được train theo ICM Delta * 10.0!)
        let features = env.state.encode_features(hero_seat);
        let raw_adv = adv_net.forward(&features);
        let max_tactical_adv = raw_adv.iter().fold(0.0f32, |acc, &v| acc.max(v));

        base_ev + (max_tactical_adv * 0.15)
    }

    /// Đệ quy tìm kiếm và cập nhật Regrets / Values
    fn traverse_subgame<R: Rng>(
        &self,
        env: &mut Environment,
        depth: usize,
        hero_seat: usize,
        initial_hero_stack: u32,
        hero_start_icm: f32,
        adv_net: &MLP,
        strat_net: Option<&MLP>,
        root_action_values: &mut [f32; NUM_ACTIONS],
        root_regrets: &mut [f32; NUM_ACTIONS],
        root_policy_sum: &mut [f32; NUM_ACTIONS],
        rng: &mut R,
    ) -> f32 {
        // 1. Nút kết thúc tự nhiên (Showdown hoặc chỉ còn 1 người)
        if env.state.street == Street::Showdown {
            return self.compute_terminal_payoff(env, hero_seat, initial_hero_stack, hero_start_icm);
        }

        let in_hand_count = env.state.players.iter().filter(|p| p.is_in_hand()).count();
        if in_hand_count <= 1 {
            return self.compute_terminal_payoff(env, hero_seat, initial_hero_stack, hero_start_icm);
        }

        // 2. Đạt độ sâu giới hạn -> Định giá nút lá
        if depth >= self.config.max_depth {
            return self.evaluate_leaf(env, hero_seat, initial_hero_stack, hero_start_icm, adv_net);
        }

        let cur = env.state.current_player_idx;
        let mask = env.get_action_mask();
        if mask.0 == 0 {
            return self.evaluate_leaf(env, hero_seat, initial_hero_stack, hero_start_icm, adv_net);
        }

        // 3. LƯỢT CỦA HERO (TRAVERSER)
        if cur == hero_seat {
            if depth == 0 {
                // TẠI ROOT NODE: Khám phá toàn bộ các nhánh hành động hợp lệ để tính chính xác EV
                let valid_actions = mask.valid_actions();
                let strat = DeepCFRPolicy::compute_regret_matching_probs(root_regrets, &mask);

                let mut current_iteration_values = [0.0f32; NUM_ACTIONS];
                let mut expected_v = 0.0f32;

                for &a in &valid_actions {
                    let a_idx = a.to_index();
                    let mut branch = env.clone();
                    let is_hand_done = branch.step(a).unwrap_or(true);
                    let v = if is_hand_done {
                        self.compute_terminal_payoff(&branch, hero_seat, initial_hero_stack, hero_start_icm)
                    } else {
                        self.traverse_subgame(
                            &mut branch,
                            depth + 1,
                            hero_seat,
                            initial_hero_stack,
                            hero_start_icm,
                            adv_net,
                            strat_net,
                            root_action_values,
                            root_regrets,
                            root_policy_sum,
                            rng,
                        )
                    };
                    current_iteration_values[a_idx] = v;
                    root_action_values[a_idx] += v;
                    expected_v += strat[a_idx] * v;
                }

                // Cập nhật Regret và Chiến thuật tích lũy
                for &a in &valid_actions {
                    let a_idx = a.to_index();
                    root_regrets[a_idx] += current_iteration_values[a_idx] - expected_v;
                    root_policy_sum[a_idx] += strat[a_idx];
                }

                expected_v
            } else {
                // Hero ở các tầng sâu hơn trong cây con:
                // Sử dụng mạng Advantage Net để chọn hành động mô phỏng
                let features = env.state.encode_features(hero_seat);
                let raw_adv = adv_net.forward(&features);
                let strat = DeepCFRPolicy::compute_regret_matching_probs(&raw_adv, &mask);
                let a = sample_action_from_probs(&strat, &mask, rng);

                let mut branch = env.clone();
                let is_hand_done = branch.step(a).unwrap_or(true);
                if is_hand_done {
                    self.compute_terminal_payoff(&branch, hero_seat, initial_hero_stack, hero_start_icm)
                } else {
                    self.traverse_subgame(
                        &mut branch,
                        depth + 1,
                        hero_seat,
                        initial_hero_stack,
                        hero_start_icm,
                        adv_net,
                        strat_net,
                        root_action_values,
                        root_regrets,
                        root_policy_sum,
                        rng,
                    )
                }
            }
        } else {
            // 4. LƯỢT CỦA ĐỐI THỦ (OPPONENT NODES)
            // Đối thủ hành động theo Blueprint Model (Strategy Net hoặc Advantage Net)
            let opp_features = env.state.encode_features(cur);
            let probs = if let Some(strat_mlp) = strat_net {
                let raw_strat = strat_mlp.forward(&opp_features);
                DeepCFRPolicy::compute_strategy_probs(&raw_strat, &mask)
            } else {
                let raw_adv = adv_net.forward(&opp_features);
                DeepCFRPolicy::compute_regret_matching_probs(&raw_adv, &mask)
            };

            let a_opp = sample_action_from_probs(&probs, &mask, rng);
            let mut branch = env.clone();
            let is_hand_done = branch.step(a_opp).unwrap_or(true);
            if is_hand_done {
                self.compute_terminal_payoff(&branch, hero_seat, initial_hero_stack, hero_start_icm)
            } else {
                self.traverse_subgame(
                    &mut branch,
                    depth + 1,
                    hero_seat,
                    initial_hero_stack,
                    hero_start_icm,
                    adv_net,
                    strat_net,
                    root_action_values,
                    root_regrets,
                    root_policy_sum,
                    rng,
                )
            }
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
    ) -> SubgameResult {
        let start_time = Instant::now();
        let mask = env.get_action_mask();
        let valid_actions = mask.valid_actions();

        // Trường hợp đặc biệt: Chỉ có 1 action duy nhất (ví dụ chỉ có thể Check)
        if valid_actions.len() <= 1 {
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
            };
        }

        let initial_hero_stack = env.state.players[hero_seat].stack;
        let start_stacks: Vec<f32> = env.state.players.iter().map(|p| (p.stack + p.current_bet) as f32).collect();
        let payouts = get_sng_payouts(start_stacks.len());
        let hero_start_icm = if self.config.use_icm && start_stacks.len() >= 2 {
            calculate_icm_ev(&start_stacks, payouts).get(hero_seat).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        let mut root_action_values = [0.0f32; NUM_ACTIONS];
        let mut root_regrets = [0.0f32; NUM_ACTIONS];
        let mut root_policy_sum = [0.0f32; NUM_ACTIONS];

        // Khởi tạo regret ban đầu từ Advantage Net để đẩy nhanh hội tụ (Warm Start)
        let root_features = env.state.encode_features(hero_seat);
        let raw_warm_adv = adv_net.forward(&root_features);
        for &a in &valid_actions {
            let i = a.to_index();
            root_regrets[i] = raw_warm_adv[i].max(0.0) * 0.5;
        }

        let mut rng = thread_rng();
        let iters = self.config.iterations;

        // 🟢 MONTE CARLO CARD SAMPLING SETUP:
        // Đánh dấu các quân bài đã biết (Dead Cards): Bài của Hero, bài chung, và bài của đối thủ nếu đã biết
        let mut dead_mask = [false; 52];
        for card in env.state.players[hero_seat].hole_cards.iter().flatten() {
            dead_mask[card.0 as usize] = true;
        }
        for card in &env.state.community_cards {
            dead_mask[card.0 as usize] = true;
        }
        for (i, p) in env.state.players.iter().enumerate() {
            if i != hero_seat {
                for card in p.hole_cards.iter().flatten() {
                    dead_mask[card.0 as usize] = true;
                }
            }
        }

        let mut alive_cards: Vec<Card> = (0..52)
            .filter(|&idx| !dead_mask[idx])
            .map(|idx| Card(idx as u8))
            .collect();

        let num_alive = env.state.players.iter().filter(|p| p.stack > 0 || p.is_all_in).count();

        for _ in 0..iters {
            let mut branch = env.clone();

            // 🟢 MONTE CARLO CARD SAMPLING:
            // Xáo trộn các lá bài chưa biết và chia cho các đối thủ còn trong ván
            for i in (1..alive_cards.len()).rev() {
                let j = rng.gen_range(0..=i);
                alive_cards.swap(i, j);
            }

            let mut deal_idx = 0;
            for (idx, p) in branch.state.players.iter_mut().enumerate() {
                if idx != hero_seat && p.is_in_hand() {
                    // Nếu đối thủ chưa có bài tẩy (None), chia 2 lá từ cọc bài
                    if p.hole_cards[0].is_none() || p.hole_cards[1].is_none() {
                        if deal_idx + 2 <= alive_cards.len() {
                            let mut c0 = alive_cards[deal_idx];
                            let mut c1 = alive_cards[deal_idx + 1];
                            deal_idx += 2;

                            // 🛡️ LỌC RANGE THỰC TẾ KẾT HỢP CẢ BÀI TẨY VÀ BÀI CHUNG (BOARD TEXTURE):
                            // Khắc phục triệt để nhược điểm của Chen Score thuần:
                            // "Có khi đối thủ cầm bài rác preflop mà trúng bài chung thì vẫn win!"
                            // -> Nếu bài rác trúng bài chung (One Pair, Two Pair, Trips, Straight, Flush): GIỮ LẠI 100%!
                            // -> Đồng thời vẫn giữ ~25% tần suất Bluff tự nhiên của đối thủ!
                            if (env.state.community_cards.len() >= 3 || num_alive > 2)
                                && (p.current_bet > env.state.bb_size || p.total_contributed > env.state.bb_size)
                            {
                                let mut attempts = 0;
                                while !is_plausible_opponent_hand(c0, c1, &env.state.community_cards, &mut rng)
                                    && deal_idx + 2 <= alive_cards.len()
                                    && attempts < 5
                                {
                                    c0 = alive_cards[deal_idx];
                                    c1 = alive_cards[deal_idx + 1];
                                    deal_idx += 2;
                                    attempts += 1;
                                }
                            }

                            p.hole_cards = [Some(c0), Some(c1)];
                        }
                    }
                }
            }

            // Nạp các lá bài còn lại vào cọc bài để chia hợp lệ nếu ván bài tiến triển qua Flop/Turn/River
            branch.deck = Deck::from_remaining(&alive_cards[deal_idx..]);

            self.traverse_subgame(
                &mut branch,
                0,
                hero_seat,
                initial_hero_stack,
                hero_start_icm,
                adv_net,
                strat_net,
                &mut root_action_values,
                &mut root_regrets,
                &mut root_policy_sum,
                &mut rng,
            );
        }

        // Tính toán EV trung bình cho từng action (tính bằng BB)
        let mut action_evs = [f32::NEG_INFINITY; NUM_ACTIONS];
        for &a in &valid_actions {
            let i = a.to_index();
            action_evs[i] = root_action_values[i] / (iters as f32);
        }

        // Tính xác suất hội tụ (Average Strategy)
        let mut action_probs = [0.0f32; NUM_ACTIONS];
        let total_policy: f32 = valid_actions.iter().map(|a| root_policy_sum[a.to_index()]).sum();
        if total_policy > 1e-4 {
            for &a in &valid_actions {
                let i = a.to_index();
                action_probs[i] = root_policy_sum[i] / total_policy;
            }
        } else {
            // Fallback Regret Matching
            action_probs = DeepCFRPolicy::compute_regret_matching_probs(&root_regrets, &mask);
        }

        // Chọn Action tốt nhất: Ưu tiên Action có EV cao nhất trong các nhánh chiến thuật GTO
        let mut best_action = valid_actions[0];
        let mut best_ev = action_evs[best_action.to_index()];

        for &a in &valid_actions[1..] {
            let ev = action_evs[a.to_index()];
            if ev > best_ev {
                best_ev = ev;
                best_action = a;
            }
        }

        let duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        SubgameResult {
            best_action,
            action_probs,
            action_evs,
            trigger,
            iterations_run: iters,
            duration_ms,
        }
    }
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
    use crate::poker_core::{Card, Rank, Suit};

    #[test]
    fn test_subgame_solver_nuts_facing_check() {
        let n = 2;
        let stacks = vec![2000, 2000];
        let mut env = Environment::new(n, &stacks, 20, 10, 0);

        // Board: Ah Kh Qh Jh 2c (Royal flush draw / Nut straight)
        env.state.community_cards = vec![
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Queen, Suit::Hearts),
            Card::new(Rank::Jack, Suit::Hearts),
            Card::new(Rank::Two, Suit::Clubs),
        ];
        env.state.street = Street::River;
        env.state.pot = 400; // 20 BB
        env.state.current_player_idx = 0; // Hero

        // Hero holds Th 9h (Royal Flush!)
        env.state.players[0].hole_cards = [
            Some(Card::new(Rank::Ten, Suit::Hearts)),
            Some(Card::new(Rank::Nine, Suit::Hearts)),
        ];
        // Opponent
        env.state.players[1].hole_cards = [
            Some(Card::new(Rank::Two, Suit::Diamonds)),
            Some(Card::new(Rank::Three, Suit::Diamonds)),
        ];

        let adv_mlp = MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]);
        let solver = SubgameSolver::with_default_config();

        let trigger = SearchTrigger::HighStakesPot { pot_bb: 20.0 };
        let result = solver.resolve(&env, 0, &adv_mlp, None, trigger);

        println!("Nuts Solver Result: Best Action = {:?}", result.best_action);
        println!("EVs = {:?}", result.action_evs);
        println!("Time = {:.2} ms", result.duration_ms);

        // Với bài Royal Flush (Nuts) ở River, Bet/Raise phải có EV dương vượt trội
        let bet_ev = result.action_evs[Action::BetPot75.to_index()]
            .max(result.action_evs[Action::BetPot33.to_index()]);
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

        // Board: Ah Kd Qs 4c 5d (Ace-King board at River)
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
        env.state.players[1].current_bet = 1000; // Opponent shove All-in 50 BB
        env.state.players[1].total_contributed = 1000;
        env.state.players[1].hole_cards = [
            Some(Card::new(Rank::Ace, Suit::Spades)),
            Some(Card::new(Rank::King, Suit::Clubs)),
        ];
        env.state.current_player_idx = 0; // Hero

        // Hero holds pure trash: 7c 2d (0% equity vs AsKc Two Pair)
        env.state.players[0].hole_cards = [
            Some(Card::new(Rank::Seven, Suit::Clubs)),
            Some(Card::new(Rank::Two, Suit::Diamonds)),
        ];

        let adv_mlp = MLP::new(&[crate::environment::FEATURE_DIM, 64, NUM_ACTIONS]);
        let solver = SubgameSolver::with_default_config();

        let trigger = SearchTrigger::UnusualOverbet { bet_pot_ratio: 2.0 };
        let result = solver.resolve(&env, 0, &adv_mlp, None, trigger);

        println!("Air Solver Result: Best Action = {:?}", result.best_action);
        println!("EVs = {:?}", result.action_evs);

        // Cầm bài 72o đối mặt All-in 50 BB -> Fold EV (0 BB) phải cao hơn Call EV (thua tiền)
        let fold_ev = result.action_evs[Action::Fold.to_index()];
        let call_ev = result.action_evs[Action::CheckCall.to_index()];
        assert!(fold_ev >= call_ev, "Cầm bài rác 72o đối mặt Shove thì Fold EV phải cao hơn Call EV!");
    }

    #[test]
    fn test_subgame_solver_third_pair_river_sampling_blocks_suicidal_allin() {
        let n = 2;
        let stacks = vec![1448, 2732];
        let mut env = Environment::new(n, &stacks, 40, 20, 0);

        // River board: 3h 5h 6c 8d Kh (tình huống thực tế Hand 2)
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
        env.state.current_player_idx = 0; // Hero

        // Hero cầm 4c 5s (Đôi 5)
        env.state.players[0].hole_cards = [
            Some(Card::new(Rank::Four, Suit::Clubs)),
            Some(Card::new(Rank::Five, Suit::Spades)),
        ];
        // Đối thủ không biết bài [None, None] -> Solver sẽ Monte Carlo Card Sampling
        env.state.players[1].hole_cards = [None, None];

        let (adv_mlp, strat_mlp) = if let (Ok(adv), Ok(strat)) = (
            MLP::load_from_file("deep_cfr_model.json"),
            MLP::load_from_file("deep_cfr_strategy.json"),
        ) {
            if adv.input_dim() <= crate::environment::FEATURE_DIM {
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
        let result = solver.resolve(&env, 0, &adv_mlp, strat_mlp.as_ref(), trigger);

        println!("Third Pair Solver Result: Best Action = {:?}", result.best_action);
        println!("EVs = {:?}", result.action_evs);

        // Với bài đôi 5 trên board có King và thùng, Solver không được phép All-In tự sát (overbet 400% pot)
        let non_allin_ev = result.action_evs[Action::BetPot33.to_index()]
            .max(result.action_evs[Action::CheckCall.to_index()])
            .max(result.action_evs[Action::BetSmall.to_index()]);
        let allin_ev = result.action_evs[Action::AllIn.to_index()];
        assert!(non_allin_ev > allin_ev, "Cầm đôi 5 trên river board K, Action có kiểm soát EV ({}) phải cao hơn AllIn EV ({})!", non_allin_ev, allin_ev);
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
        env.state.current_player_idx = 1; // Hero (Ghế #1, stack 2498)
        env.state.players[0].current_bet = 40;
        env.state.players[2].current_bet = 20;
        env.state.players[3].current_bet = 40;

        // Hero holds [T♣ 2♠] (pure air)
        env.state.players[1].hole_cards = [
            Some(Card::new(Rank::Ten, Suit::Clubs)),
            Some(Card::new(Rank::Two, Suit::Spades)),
        ];
        // Opponents hold [None, None]
        env.state.players[0].hole_cards = [None, None];
        env.state.players[2].hole_cards = [None, None];
        env.state.players[3].hole_cards = [None, None];

        let (adv_mlp, strat_mlp) = if let (Ok(adv), Ok(strat)) = (
            MLP::load_from_file("deep_cfr_model.json"),
            MLP::load_from_file("deep_cfr_strategy.json"),
        ) {
            if adv.input_dim() <= crate::environment::FEATURE_DIM {
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
        let result = solver.resolve(&env, 1, &adv_mlp, strat_mlp.as_ref(), trigger);

        println!("Preflop Trash Solver Result: Best Action = {:?}", result.best_action);
        println!("EVs = {:?}", result.action_evs);

        // Với T2o ở Preflop đối mặt cú bet 40 chip, không được phép coi BetPot75 hay AllIn là tối ưu!
        let fold_ev = result.action_evs[Action::Fold.to_index()];
        let bet75_ev = result.action_evs[Action::BetPot75.to_index()];
        let allin_ev = result.action_evs[Action::AllIn.to_index()];
        println!("Fold EV: {}, Bet75 EV: {}, AllIn EV: {}", fold_ev, bet75_ev, allin_ev);
        assert!(fold_ev >= bet75_ev || result.best_action != Action::BetPot75, "T2o không được phép coi BetPot75 là tối ưu áp đảo!");
        assert_ne!(result.best_action, Action::AllIn, "T2o không được All-In!");
    }

    #[test]
    fn test_subgame_solver_heads_up_duel_auto_trigger() {
        let n = 2;
        let stacks = vec![3000, 1000]; // Hero chipleader 3000 vs Opponent 1000
        let mut env = Environment::new(n, &stacks, 40, 20, 0);
        env.state.street = Street::Preflop;
        env.state.pot = 60;
        env.state.current_player_idx = 0; // Hero on Button (SB)

        let solver = SubgameSolver::with_default_config();
        // Dù force_search = false, Solver vẫn PHẢI tự động kích hoạt HeadsUpDuel!
        let trigger = solver.should_trigger_search(&env, 0, false);
        assert_eq!(trigger, SearchTrigger::HeadsUpDuel);
    }

    #[test]
    fn test_subgame_solver_multiway_folded_to_two_does_not_trigger_heads_up_duel() {
        let n = 8;
        let stacks = vec![1000; 8];
        let mut env = Environment::new(n, &stacks, 40, 20, 0);
        env.state.street = Street::Preflop;
        env.state.pot = 60; // Pot nhỏ 1.5 BB
        // 6 người fold preflop
        for i in 2..8 {
            env.state.players[i].is_folded = true;
        }
        env.state.current_player_idx = 0;

        let solver = SubgameSolver::with_default_config();
        // Bàn 8 người còn sống trong tour, dù chỉ 2 người chưa fold trong ván này: KHÔNG được kích hoạt HeadsUpDuel!
        let trigger = solver.should_trigger_search(&env, 0, false);
        assert_eq!(trigger, SearchTrigger::FastPolicyFallback);
    }

    #[test]
    fn test_is_plausible_opponent_hand_preserves_trash_that_hits_board() {
        let mut rng = thread_rng();
        // 1. Bài rác 8c 3d (Chen cực thấp < 0.10)
        let c0 = Card::from_str_repr("8c").unwrap();
        let c1 = Card::from_str_repr("3d").unwrap();

        // Flop ra 8h 3s 2d -> Đối thủ trúng Two Pair từ bài rác!
        let board_hit = vec![
            Card::from_str_repr("8h").unwrap(),
            Card::from_str_repr("3s").unwrap(),
            Card::from_str_repr("2d").unwrap(),
        ];
        // Bắt buộc phải GIỮ LẠI 100% (true)!
        assert!(
            is_plausible_opponent_hand(c0, c1, &board_hit, &mut rng),
            "Bài rác 83o trúng Two Pair trên board 8-3-2 phải được giữ lại 100%!"
        );

        // 2. Bài rác 7c 2d trúng Trips trên 7h 7s Kd
        let c_72_0 = Card::from_str_repr("7c").unwrap();
        let c_72_1 = Card::from_str_repr("2d").unwrap();
        let board_trips = vec![
            Card::from_str_repr("7h").unwrap(),
            Card::from_str_repr("7s").unwrap(),
            Card::from_str_repr("Kd").unwrap(),
        ];
        assert!(
            is_plausible_opponent_hand(c_72_0, c_72_1, &board_trips, &mut rng),
            "Bài rác 72o trúng Trips trên board 7-7-K phải được giữ lại 100%!"
        );
    }
}

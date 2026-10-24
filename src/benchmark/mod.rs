//! Benchmark: đo sức mạnh policy bằng bb/100 so với các bot baseline và bàn hỗn hợp.

use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::self_play::{Policy, UniformRandomPolicy};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::io::Write;

pub mod arena;
pub use arena::{
    evaluate_against_champion, evaluate_against_champion_cash, evaluate_against_champion_final_table,
    evaluate_against_champion_table, evaluate_against_champion_tournament, evaluate_against_heuristic_team,
    evaluate_arena_suite, print_arena_results, simulate_demo_tournament, simulate_hero_tournament,
    ArenaMatchResult, TournamentArenaResult,
};

// ---- Vị trí các đặc trưng trong encode_features (đổi layout thì phải cập nhật; có test kiểm tra) ----
const F_POS_BTN: usize = 109;
const F_POS_SB: usize = 110;
#[allow(dead_code)]
const F_POS_BB: usize = 111;
const F_POS_CO: usize = 116;
const F_IS_BUBBLE: usize = 119;     // 1.0 nếu còn 5 người (vòng Bubble)
const F_STACK_RANK: usize = 121;    // 0.0 (nghèo nhất) .. 1.0 (Chip Leader)
const F_STREET_PREFLOP: usize = 123;
#[allow(dead_code)]
const F_STREET_FLOP: usize = 124;
const F_STREET_TURN: usize = 125;
const F_STREET_RIVER: usize = 126;
const F_TO_CALL_NORM: usize = 128; // to_call / (20 * BB) -> 1 BB = 0.05
const F_POT_ODDS: usize = 130;
const F_IN_HAND: usize = 160;
const F_EFF_BB: usize = 161;        // eff_stack_bb / 50.0
const F_PUSHFOLD: usize = 162;
const F_TO_CALL_RATIO_STACK: usize = 163;
const F_POT_COMMITMENT: usize = 164;
const F_HAND_CAT: usize = 165;
const F_HAND_STRENGTH: usize = 166; // preflop: điểm Chen, postflop: Monte Carlo current hand strength (0..1)
const F_EQUITY: usize = 167;
const F_AGG_BARRELS: usize = 169;   // Số vòng đối thủ đã barrel / 4.0
const F_FLUSH_DRAW: usize = 170;
const F_STRAIGHT_DRAW: usize = 171;
const F_PAIRED_BOARD: usize = 172;
const F_MAX_SUIT_BOARD: usize = 173;
const F_BOARD_HIGH: usize = 174;
#[allow(dead_code)]
const F_FACING_RAISE: usize = 175;
#[allow(dead_code)]
const F_RELATIVE_BET_SIZE: usize = 176;

/// Bộ sinh số giả ngẫu nhiên thuần túy (Deterministic Randomizer) từ đặc trưng ván bài
/// giúp bot thực hiện các chiến thuật hỗn hợp (GTO Mixing / Trapping) mà không làm mất tính tái lập.
#[inline(always)]
fn pseudo_rand(f: &[f32]) -> f32 {
    let s = f.iter().take(20).enumerate().fold(0.0f32, |acc, (i, &v)| acc + (i as f32 + 1.0) * v);
    ((s * 1234.567 + 0.123).fract()).abs()
}

/// Ở Preflop, chỉ coi là đang đối mặt với Bet/Raise nếu to_call > 1 BB (đã có người Open-Raise).
/// Nếu to_call <= 1 BB thì là Unopened Pot -> Quân xanh phải chủ động Open-Raise bài mạnh hoặc Fold bài rác!
#[inline(always)]
fn is_facing_real_bet(f: &[f32]) -> bool {
    if f[F_STREET_PREFLOP] > 0.5 {
        f[F_TO_CALL_NORM] > 0.055
    } else {
        f[F_TO_CALL_NORM] > 0.001
    }
}

/// Quy đổi Equity sang thang đo Edge đồng nhất cho mọi quy mô bàn (từ Heads-Up 2 người đến 8-Max):
/// Khắc phục triệt để lỗi `eq * 2.0 <= 2.0` làm tê liệt các ngưỡng > 1.75..2.35 khi còn 2-3 người!
#[inline(always)]
fn normalized_edge(eq: f32, n: f32) -> f32 {
    if n <= 2.5 {
        // Bàn 2 người (Heads-Up): eq = 0.50 -> edge = 1.00; eq = 0.65 (AQo) -> 1.72; eq = 0.85 (AA) -> 2.68; eq = 1.0 -> 3.40
        1.0 + (eq - 0.50) * 4.8
    } else if n <= 3.5 {
        // Bàn 3 người (3-Max): eq = 0.333 -> edge = 1.00; eq = 0.73 (AA) -> 2.59
        1.0 + (eq - 0.333) * 4.0
    } else {
        eq * n
    }
}

/// Ngưỡng Chen tối thiểu để bắt chết cú All-in / Re-shove láo ở Preflop theo số người trong ván:
/// - Bàn 2 người (Heads-Up): chen >= 0.27 (22+, A2+, K7+, Q9+, J9s - Top 32% bài) là bắt chết kẻ shove 3-6!
/// - Bàn 3-4 người: chen >= 0.35 (55+, A7+, KTs+, QJs)
/// - Bàn 5-8 người: chen >= 0.42 (88+, ATs+, AJo+, KQo)
#[inline(always)]
fn anti_shove_chen_threshold(n: f32) -> f32 {
    if n <= 2.5 {
        0.27
    } else if n <= 4.5 {
        0.35
    } else {
        0.42
    }
}

/// Ngưỡng xác định bài rác Preflop thích ứng theo quy mô bàn (Heads-Up vs Multiway):
/// - Bàn đông người (5 - 8 người): Chen < 0.22 là rác (như 72o, 83o, 94o, K4o).
/// - Bàn 3 - 4 người (Short-handed): Chen < 0.16 là rác.
/// - Bàn Heads-Up (2 người): Đối đầu 1-vs-1, K-high/Q-high có equity > 50%, chỉ Chen < 0.10 (72o, 82o, 42o, 32o) mới thực sự là rác!
#[inline]
pub fn trash_chen_threshold(in_hand: usize) -> f32 {
    if in_hand <= 2 {
        0.10
    } else if in_hand <= 4 {
        0.16
    } else {
        0.22
    }
}
pub const TRASH_CHEN: f32 = 0.22;
const TABLE_SIZE: usize = 8;

// ===================================================================== Baselines

fn pick(mask: &ActionMask, prefs: &[Action]) -> Action {
    prefs
        .iter()
        .copied()
        .find(|&a| mask.is_valid(a))
        .or_else(|| mask.valid_actions().first().copied())
        .unwrap_or(Action::Fold)
}

fn one_hot(a: Action) -> [f32; NUM_ACTIONS] {
    let mut p = [0.0; NUM_ACTIONS];
    p[a.to_index()] = 1.0;
    p
}

/// Luôn check nếu miễn phí, nếu bị bet thì fold (Mốc tham chiếu: người chơi hoàn toàn thụ động)
pub struct AlwaysFold;
impl Policy for AlwaysFold {
    fn select_action(&self, _f: &[f32], m: &ActionMask) -> Action {
        // Nếu có thể Check (không tốn chip) thì check, ngược lại fold
        if m.is_valid(Action::CheckCall) && !m.is_valid(Action::Fold) {
            Action::CheckCall
        } else {
            pick(m, &[Action::Fold, Action::CheckCall])
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// Luôn check/call
pub struct CallStation;
impl Policy for CallStation {
    fn select_action(&self, _f: &[f32], m: &ActionMask) -> Action {
        pick(m, &[Action::CheckCall, Action::Fold])
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// Luôn all-in nếu được
pub struct Maniac;
impl Policy for Maniac {
    fn select_action(&self, _f: &[f32], m: &ActionMask) -> Action {
        pick(m, &[Action::AllIn, Action::CheckCall])
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// Chặt-hung hăng dựa trên equity, vị trí, cấu trúc mặt bài và bảo vệ pot (TAG v2)
#[derive(Clone)]
pub struct TightAggressive;
impl Policy for TightAggressive {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let edge = normalized_edge(eq, n); // 1.0 = bài trung bình trên bàn
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let chen = f[F_HAND_STRENGTH]; // Điểm Chen ở Preflop (0..1)
        let cat = f[F_HAND_CAT];       // 0.0 = High Card, 1/9 = One Pair, 2/9 = Two Pair, 3/9 = Trips...
        let is_preflop = f[F_STREET_PREFLOP] > 0.5;
        let is_wet_board = f[F_MAX_SUIT_BOARD] >= 0.60 || f[F_STRAIGHT_DRAW] > 0.5;
        let is_dry_paired = f[F_PAIRED_BOARD] > 0.5 && f[F_MAX_SUIT_BOARD] <= 0.40;
        let is_committed = f[F_POT_COMMITMENT] > 0.38 || f[F_TO_CALL_RATIO_STACK] > 0.60;
        let rand = pseudo_rand(f);

        // 1. Chế độ Push/Fold (< 12 BB):
        if f[F_PUSHFOLD] > 0.5 {
            let push_chen = if n <= 2.5 { 0.28 } else if n <= 4.5 { 0.38 } else { 0.48 };
            if edge > 1.75 || (is_preflop && chen >= push_chen) {
                return pick(m, &[Action::AllIn, Action::CheckCall]);
            } else if facing {
                let call_chen = anti_shove_chen_threshold(n);
                if is_preflop && (chen >= call_chen || edge > 1.45) {
                    return pick(m, &[Action::CheckCall, Action::AllIn]);
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }
        }

        // 2. Pot Commitment: Đã bỏ nhiều chip vào pot và có bài Made Hand -> Đẩy All-In không tiếc!
        if is_committed && (cat >= 0.11 || edge > 1.50) {
            return pick(m, &[Action::AllIn, Action::BetPot75, Action::CheckCall]);
        }

        // 3. Khi đối mặt với Bet / Raise / All-In của đối thủ:
        if facing {
            // A. Chống Shove Preflop:
            if is_preflop {
                let min_chen = anti_shove_chen_threshold(n);
                if chen >= min_chen || edge > 1.45 {
                    if chen >= (min_chen + 0.18).min(0.60) || edge > 2.0 {
                        return pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall]);
                    } else {
                        return pick(m, &[Action::CheckCall]);
                    }
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }

            // B. Postflop đối mặt Bet/Raise:
            // - Cầm Monster Hand (Set/Two Pair lớn/Straight/Flush, cat >= 2/9 hoặc edge > 2.2):
            //   Check-Raise / 3-Bet to (BetPot75 / AllIn) với 80% tần suất để lấy max value, 20% trap Call!
            if (cat >= 0.22 || edge > 2.20) && rand < 0.80 {
                return pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall]);
            }

            // - Cầm bài rất mạnh (Top Pair kicker xịn hoặc Overpair, edge > 1.85):
            //   Nếu mặt bài ướt, Raise BetPot75 để bảo vệ bài, nếu không thì Call.
            if edge > 1.85 && is_wet_board && rand < 0.60 {
                return pick(m, &[Action::BetPot75, Action::CheckCall]);
            }

            // - Call chuẩn toán học Pot Odds + Margin an toàn:
            if eq > pot_odds + 0.03 || edge > 1.50 || (cat >= 0.11 && eq > pot_odds - 0.02) {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        } else {
            // 4. Khi chưa có ai bet (Unopened Pot / Checked to us):
            if is_preflop {
                let is_late = f[F_POS_BTN] > 0.5 || f[F_POS_CO] > 0.5 || f[F_POS_SB] > 0.5;
                let open_chen = if is_late { 0.26 } else { 0.32 };
                if chen >= open_chen || edge > 1.40 {
                    // Open-raise chuẩn 33% hoặc 75%
                    pick(m, &[Action::BetPot33, Action::BetPot75, Action::CheckCall])
                } else if f[F_TO_CALL_NORM] > 0.05 {
                    pick(m, &[Action::Fold, Action::CheckCall])
                } else {
                    pick(m, &[Action::CheckCall])
                }
            } else {
                // Postflop:
                // Monster: Bet to (BetPot75) hoặc Trap Check (15% mixing)
                if cat >= 0.22 || edge > 2.0 {
                    if rand < 0.85 {
                        pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
                    } else {
                        pick(m, &[Action::CheckCall]) // Trap slowplay
                    }
                } else if edge > 1.50 || (cat >= 0.11 && is_wet_board) {
                    // Top Pair trên Wet board -> Bet to (BetPot75) để bảo vệ (Protection)
                    if is_wet_board {
                        pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
                    } else {
                        pick(m, &[Action::BetPot33, Action::BetSmall, Action::CheckCall])
                    }
                } else if edge > 1.25 {
                    // C-bet chuẩn trên Dry board
                    if is_dry_paired {
                        pick(m, &[Action::BetSmall, Action::BetPot33, Action::CheckCall])
                    } else {
                        pick(m, &[Action::BetPot33, Action::CheckCall])
                    }
                } else if edge > 1.05 && rand < 0.40 {
                    // Probe bet nhỏ 15% thăm dò
                    pick(m, &[Action::BetSmall, Action::CheckCall])
                } else {
                    // Pot control với bài trung bình yếu
                    pick(m, &[Action::CheckCall])
                }
            }
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 1. LAG Shark (Loose-Aggressive v2):
/// - Tận dụng vị trí cuối (BTN, CO) để mở rộng range tấn công và Steal blinds.
/// - Biết Semi-bluff Raise khi có Flush Draw hoặc Straight Draw ở Flop/Turn.
/// - Tấn công Double-barrel Turn khi xuất hiện Scare Card (A, K) trên board.
#[derive(Clone)]
pub struct LagShark;
impl Policy for LagShark {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let is_late_pos = f[F_POS_BTN] > 0.5 || f[F_POS_CO] > 0.5;
        let has_draw = f[F_FLUSH_DRAW] > 0.5 || f[F_STRAIGHT_DRAW] > 0.5;
        let chen = f[F_HAND_STRENGTH];
        let cat = f[F_HAND_CAT];
        let is_preflop = f[F_STREET_PREFLOP] > 0.5;
        let is_turn = f[F_STREET_TURN] > 0.5;
        let is_river = f[F_STREET_RIVER] > 0.5;
        let board_has_scare = f[F_BOARD_HIGH] >= 0.83; // A hoặc K trên board
        let rand = pseudo_rand(f);

        // Cộng thêm Equity kỳ vọng khi có draw hoặc lợi thế vị trí
        let draw_bonus = if has_draw && !is_river { 0.14 } else { 0.0 };
        let pos_bonus = if is_late_pos && !facing { 0.10 } else { 0.0 };
        let eff_eq = (eq + draw_bonus + pos_bonus).min(0.98);
        let edge = normalized_edge(eff_eq, n);

        let min_chen = anti_shove_chen_threshold(n);
        let push_edge = if n <= 2.5 { 1.65 } else { 2.05 };
        if f[F_PUSHFOLD] > 0.5 && (edge > push_edge || (is_preflop && chen >= min_chen)) {
            return pick(m, &[Action::AllIn, Action::CheckCall]);
        }

        if facing {
            // 1. Preflop đối mặt Bet/Raise:
            if is_preflop {
                if chen >= min_chen || edge > 1.45 {
                    return pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall]);
                }
                // 3-Bet Light khi ở BTN với bài có kết nối (Suited / Connector Chen >= 0.25):
                if is_late_pos && chen >= 0.25 && rand < 0.35 {
                    return pick(m, &[Action::BetPot75, Action::CheckCall]);
                }
                if eq >= pot_odds - 0.02 && chen >= 0.22 {
                    return pick(m, &[Action::CheckCall]);
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }

            // 2. Postflop đối mặt Bet:
            // Semi-Bluff Raise khi có Monster Draw (Flush/Straight Draw):
            if has_draw && !is_river && rand < 0.45 {
                return pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall]);
            }

            if edge > 2.10 || cat >= 0.22 {
                // Value Raise to
                pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
            } else if eff_eq > pot_odds + 0.01 || edge > 1.48 {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        } else {
            // 3. Khi chưa ai Bet:
            if is_preflop {
                // Steal Blind mạnh mẽ ở Late Position
                let steal_chen = if is_late_pos { 0.20 } else { 0.28 };
                if chen >= steal_chen || edge > 1.20 {
                    pick(m, &[Action::BetPot33, Action::BetPot75, Action::CheckCall])
                } else if f[F_TO_CALL_NORM] > 0.05 {
                    pick(m, &[Action::Fold, Action::CheckCall])
                } else {
                    pick(m, &[Action::CheckCall])
                }
            } else {
                // Postflop:
                // Double Barrel ở Turn với Scare Card (A hoặc K trên bàn) khi đối thủ vừa Check
                if is_turn && board_has_scare && rand < 0.65 {
                    pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
                } else if edge > 1.70 || (has_draw && edge > 1.20) {
                    pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
                } else if edge > 1.25 || (is_late_pos && edge > 0.95) {
                    // C-bet hung hăng
                    pick(m, &[Action::BetPot33, Action::BetSmall, Action::CheckCall])
                } else if edge > 1.00 && !is_river {
                    pick(m, &[Action::BetSmall, Action::CheckCall])
                } else {
                    pick(m, &[Action::CheckCall])
                }
            }
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 2. NashPushBot (Chuyên gia Short-Stack & Push/Fold v2):
/// - Phân hóa chi tiết 5 bậc vị trí theo lý thuyết Nash Equilibrium (SB vs BB > BTN > CO > MP > UTG).
/// - Áp dụng Gap Concept của Sklansky khi Call Shove.
/// - Khi stack sâu (> 15 BB): Đánh TAG chuẩn mực.
#[derive(Clone)]
pub struct NashPushBot;
impl Policy for NashPushBot {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let edge = normalized_edge(eq, n);
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let eff_bb = f[F_EFF_BB] * 50.0;
        let chen = f[F_HAND_STRENGTH];
        let is_preflop = f[F_STREET_PREFLOP] > 0.5;

        // Vị trí chi tiết theo chuẩn Nash Equilibrium
        let is_btn = f[F_POS_BTN] > 0.5;
        let is_sb = f[F_POS_SB] > 0.5;
        let is_co = f[F_POS_CO] > 0.5;
        let is_late = is_btn || is_sb || is_co;

        // Chế độ Short-stack Nash Push/Fold (<= 15 BB):
        if eff_bb <= 15.0 && is_preflop {
            let min_chen = anti_shove_chen_threshold(n);

            if !facing {
                // Open-Shove theo bậc vị trí chi tiết của Nash:
                // - SB vs BB: cực rộng (Chen >= 0.16)
                // - BTN: rộng (Chen >= 0.20)
                // - CO: vừa (Chen >= 0.26)
                // - Early (UTG): rất chặt (Chen >= 0.36)
                let (pos_chen_thresh, push_edge_thresh) = if is_sb {
                    (0.16, 0.95 + (eff_bb / 55.0))
                } else if is_btn {
                    (0.20, 1.05 + (eff_bb / 50.0))
                } else if is_co {
                    (0.26, 1.20 + (eff_bb / 45.0))
                } else if is_late {
                    (0.30, 1.35 + (eff_bb / 40.0))
                } else {
                    (0.36, 1.55 + (eff_bb / 35.0))
                };

                if chen >= pos_chen_thresh || edge >= push_edge_thresh {
                    return pick(m, &[Action::AllIn, Action::BetPot75, Action::CheckCall]);
                } else {
                    return pick(m, &[Action::Fold, Action::CheckCall]);
                }
            } else {
                // Call All-in: Áp dụng Sklansky Gap Concept (đòi hỏi range chặt hơn tự Open-Shove)
                let call_edge = if n <= 2.5 { 1.20 } else { 1.45 };
                let call_min_chen = if is_late { min_chen } else { (min_chen + 0.05).min(0.55) };

                if (eq >= pot_odds + 0.04 || chen >= call_min_chen) && edge >= call_edge {
                    return pick(m, &[Action::CheckCall, Action::AllIn]);
                } else {
                    return pick(m, &[Action::Fold, Action::CheckCall]);
                }
            }
        }

        // Stack sâu (> 15 BB): Chơi TAG chuẩn mực
        TightAggressive.select_action(f, m)
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 3. IcmSniper (TRÙM CUỐI HEURISTIC - Apex SNG Bot v2):
/// - Khấu trừ Equity thông minh khi đối thủ barrel nhiều vòng liên tục.
/// - Dynamic Risk Premium: Hạ thuế khi cô lập Short-stack, tăng thuế khi gặp Chip Leader ở Bubble.
/// - Gây áp lực cực hạn khi làm Chip Leader ở vòng Bubble.
#[derive(Clone)]
pub struct IcmSniper;
impl Policy for IcmSniper {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let raw_eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let is_late = f[F_POS_BTN] > 0.5 || f[F_POS_CO] > 0.5;
        let has_draw = (f[F_FLUSH_DRAW] > 0.5 || f[F_STRAIGHT_DRAW] > 0.5) && f[F_STREET_RIVER] < 0.5;
        let chen = f[F_HAND_STRENGTH];
        let cat = f[F_HAND_CAT];

        // 1. Phân tích lịch sử Barrel của đối thủ:
        let opp_barrels = f[F_AGG_BARRELS] * 4.0;
        let range_discount = if facing { (opp_barrels * 0.065).min(0.22) } else { 0.0 };

        // 2. Dynamic Risk Premium & Bubble Pressure:
        let is_bubble = f[F_IS_BUBBLE] > 0.5;
        let stack_rank = f[F_STACK_RANK]; // 1.0 = giàu nhất bàn
        let to_call_ratio = f[F_TO_CALL_RATIO_STACK]; // Tỉ lệ chip phải call so với stack

        let icm_tax = if is_bubble && facing {
            if stack_rank >= 0.85 {
                0.01 // Là Chip Leader ở Bubble: rủi ro thấp
            } else if stack_rank < 0.50 {
                0.08 // Stack trung bình/ngắn ở Bubble: tăng 8% equity đòi hỏi mới dám Call!
            } else {
                0.05
            }
        } else if stack_rank > 0.80 && !facing {
            -0.05 // Là Chip Leader và chưa ai bet -> giảm 5% equity để ép cả bàn!
        } else if facing && to_call_ratio > 0.70 && stack_rank > 0.60 {
            // Đối thủ short-stack shove vào mặt mình -> giảm 4% equity để cô lập diệt mồi
            -0.04
        } else {
            0.0
        };

        let draw_bonus = if has_draw { 0.10 } else { 0.0 };
        let real_eq = (raw_eq + draw_bonus - range_discount).clamp(0.0, 1.0);
        let edge = normalized_edge(real_eq, n);
        let min_chen = anti_shove_chen_threshold(n);

        // Chế độ Push/Fold (< 12 BB) có điều chỉnh ICM
        if f[F_PUSHFOLD] > 0.5 {
            let push_req = if facing {
                if n <= 2.5 { 1.40 + icm_tax * 3.0 } else { 1.80 + icm_tax * 4.0 }
            } else if is_late {
                1.15 + icm_tax
            } else {
                1.40 + icm_tax
            };
            if (!facing && edge >= push_req)
                || (facing && (edge >= push_req || chen >= min_chen) && (real_eq >= pot_odds - 0.02 + icm_tax || chen >= min_chen))
            {
                return pick(m, &[Action::AllIn, Action::CheckCall]);
            } else {
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }
        }

        if facing {
            // Preflop chống shove
            if f[F_STREET_PREFLOP] > 0.5 && (chen >= min_chen || edge > 1.45) {
                if chen >= (min_chen + 0.18).min(0.60) || edge > 2.0 {
                    return pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall]);
                } else {
                    return pick(m, &[Action::CheckCall]);
                }
            }

            // Postflop:
            if edge > 2.30 || cat >= 0.25 {
                pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall])
            } else if edge > 1.90 {
                pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
            } else if real_eq > pot_odds + 0.03 + icm_tax || edge > 1.55 {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        } else {
            // Khi chưa ai bet:
            // Chip Leader ép bàn ở Bubble hoặc bài cực mạnh
            if stack_rank > 0.80 && is_bubble {
                pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
            } else if edge > 1.95 || cat >= 0.25 {
                pick(m, &[Action::BetPot75, Action::CheckCall])
            } else if edge > 1.30 || (is_late && edge > 1.10) {
                pick(m, &[Action::BetPot33, Action::BetSmall, Action::CheckCall])
            } else if edge > 1.05 && f[F_STREET_PREFLOP] < 0.5 {
                pick(m, &[Action::BetSmall, Action::CheckCall])
            } else if f[F_STREET_PREFLOP] > 0.5 && f[F_TO_CALL_NORM] > 0.05 {
                pick(m, &[Action::Fold, Action::CheckCall])
            } else {
                pick(m, &[Action::CheckCall])
            }
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 4. NitTrap (The Rock / Trapper):
/// - Siêu chặt preflop: VPIP chỉ ~12-15% (chỉ chơi pairs, AJs+, AKo).
/// - Gài bẫy chết người: Limp-trap ở preflop hoặc Check-Call flop với Monster (Set, Overpair),
///   sau đó phản công Check-Raise to / Shove ở Turn hoặc River để gom sạch chip của kẻ cẩu thả!
#[derive(Clone)]
pub struct NitTrap;
impl Policy for NitTrap {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let edge = normalized_edge(eq, n);
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let chen = f[F_HAND_STRENGTH];
        let cat = f[F_HAND_CAT];
        let is_preflop = f[F_STREET_PREFLOP] > 0.5;
        let is_river = f[F_STREET_RIVER] > 0.5;
        let rand = pseudo_rand(f);

        // Preflop:
        if is_preflop {
            // Khi dưới 12 BB (Push/Fold): Chỉ push hoặc call với top tier bài
            if f[F_PUSHFOLD] > 0.5 {
                if chen >= 0.45 || edge > 2.0 {
                    return pick(m, &[Action::AllIn, Action::CheckCall]);
                } else if facing && (chen >= 0.40 || edge > 1.80) {
                    return pick(m, &[Action::CheckCall, Action::AllIn]);
                } else {
                    return pick(m, &[Action::Fold, Action::CheckCall]);
                }
            }

            // Bài rác hoặc trung bình (< 0.35 điểm Chen: dưới TT, AJs, KQs):
            // Nit không bao giờ dính líu, Fold ngay lập tức!
            if chen < 0.35 {
                if !facing && m.is_valid(Action::CheckCall) && !m.is_valid(Action::Fold) {
                    return Action::CheckCall; // Check miễn phí ở BB
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }

            // Cầm bài quái vật (QQ+, AKs, Chen >= 0.48):
            // 40% Limp/Flat Trap gài bẫy, 60% Open-Raise to BetPot75!
            if chen >= 0.48 && rand < 0.40 {
                return pick(m, &[Action::CheckCall, Action::BetPot75]);
            }
            return pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall]);
        }

        // Postflop:
        // Nếu không dính Two Pair trở lên (cat < 0.22) và edge không vượt trội:
        if cat < 0.22 && edge < 1.70 {
            if facing {
                if eq > pot_odds + 0.08 {
                    return pick(m, &[Action::CheckCall]);
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            } else {
                return pick(m, &[Action::CheckCall]); // Check pot control
            }
        }

        // Đang cầm Monster (Set, Two Pair lớn, Straight, Flush):
        // Flop: Check-Call gài bẫy đối thủ barrel.
        // Turn / River: Tung cú Check-Raise / Shove hủy diệt!
        if facing {
            if is_river || edge > 2.25 {
                pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall])
            } else if rand < 0.50 {
                pick(m, &[Action::CheckCall]) // Slow-play call bẫy
            } else {
                pick(m, &[Action::BetPot75, Action::CheckCall])
            }
        } else {
            if is_river {
                pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall])
            } else if rand < 0.35 {
                pick(m, &[Action::CheckCall]) // Trap check-raise
            } else {
                pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
            }
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 5. StickyFish (Inelastic Calling Station v2):
/// - Vào pot rộng preflop (VPIP 45-55%), thích xem flop với bài có kết nối hoặc suited.
/// - Postflop:
///   + Khi có bài mạnh (Two Pair, Trips, Straight, Flush... cat >= 0.20): Chủ động Bet lấy value, không check thụ động!
///   + Khi đối mặt bet vừa (to_call <= 0.75 pot): Lì lợm Call với mọi Pair và Draw (không bluff raise, không fold pair).
///   + Bản năng sinh tồn: Khi đối mặt All-In hoặc Bet cực nặng (> 0.75 pot / stack commit lớn):
///     Chỉ Call nếu có Top Pair trở lên (cat >= 0.11 và chen/equity tốt), dứt khoát Fold bài rác / bottom pair / gutshot yếu!
#[derive(Clone)]
pub struct StickyFish;
impl Policy for StickyFish {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let chen = f[F_HAND_STRENGTH];
        let cat = f[F_HAND_CAT];
        let has_draw = f[F_FLUSH_DRAW] > 0.5 || f[F_STRAIGHT_DRAW] > 0.5;
        let is_preflop = f[F_STREET_PREFLOP] > 0.5;
        let call_ratio = f[F_TO_CALL_RATIO_STACK]; // Tỷ lệ chip phải bỏ ra so với stack hiện tại
        let is_huge_bet = call_ratio > 0.35 || f[F_TO_CALL_NORM] > 0.25; // Bet > 5 BB hoặc > 35% stack / All-in

        // 1. Preflop:
        if is_preflop {
            // Cầm siêu phẩm (AA, KK, AKs, chen >= 0.50): Raise to hoặc Shove
            if chen >= 0.50 {
                return pick(m, &[Action::AllIn, Action::BetPot75, Action::CheckCall]);
            }
            // Đối mặt với Bet lớn / Shove All-In:
            if is_huge_bet {
                if chen >= 0.35 || eq >= 0.38 {
                    return pick(m, &[Action::CheckCall, Action::AllIn]);
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }
            // Thích xem Flop: Call mọi open-raise bình thường với bài có tiềm năng (chen >= 0.10 hoặc suited/connectors)
            if chen >= 0.10 || eq >= 0.20 {
                return pick(m, &[Action::CheckCall, Action::Fold]);
            }
            // Unopened pot: Limp vào xem flop
            if !facing {
                return pick(m, &[Action::CheckCall, Action::Fold]);
            }
            return pick(m, &[Action::Fold, Action::CheckCall]);
        }

        // 2. Postflop:
        // A. Cầm bài rất mạnh (Two Pair, Trips, Straight, Flush... cat >= 0.20):
        // Fish không check thụ động nữa! Chủ động Bet để lấy tiền từ đối thủ!
        if cat >= 0.20 {
            if facing {
                // Đối mặt bet: Call hoặc Raise
                return pick(m, &[Action::BetPot75, Action::CheckCall, Action::AllIn]);
            } else {
                return pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall]);
            }
        }

        // B. Cầm bài tốt (One Pair - cat >= 0.10):
        if cat >= 0.10 {
            if facing {
                // Nếu đối thủ Shove All-In hoặc Bet cực to (> 35% stack):
                // Phải có ít nhất Top Pair / bài tốt (eq > 0.45 hoặc chen > 0.30) mới call, không call tự sát với bottom pair!
                if is_huge_bet && eq < 0.45 && chen < 0.30 {
                    return pick(m, &[Action::Fold, Action::CheckCall]);
                }
                // Đối mặt bet thông thường (<= 0.75 pot): Dính One Pair là LÌ ĐÒN CALL 100%!
                return pick(m, &[Action::CheckCall, Action::Fold]);
            } else {
                // Chưa ai bet: Fish cầm Top/Mid pair thỉnh thoảng lead bet (Donk bet 33% pot hoặc Check)
                let rand = pseudo_rand(f);
                if rand < 0.35 {
                    return pick(m, &[Action::BetPot33, Action::BetSmall, Action::CheckCall]);
                }
                return pick(m, &[Action::CheckCall]);
            }
        }

        // C. Cầm sảnh/thùng chờ (Draws) hoặc có pot odds tốt:
        if has_draw || eq > (pot_odds - 0.05) {
            if facing {
                // Đối mặt bet cực to / All-in khi chỉ có Draw:
                if is_huge_bet && eq < pot_odds {
                    return pick(m, &[Action::Fold, Action::CheckCall]);
                }
                return pick(m, &[Action::CheckCall, Action::Fold]);
            } else {
                return pick(m, &[Action::CheckCall]);
            }
        }

        // D. Không có gì (Air / High Card):
        if facing {
            pick(m, &[Action::Fold, Action::CheckCall])
        } else {
            pick(m, &[Action::CheckCall])
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 6. ThreeBetAggro (Squeeze Bully / Kẻ Khủng Bố 3-Bet):
/// - Preflop: Thường xuyên 3-Bet to (BetPot75) hoặc Shove khi có người Open-Raise để cướp pot trước flop.
/// - Cực kỳ nguy hiểm khi tung cú Squeeze trừng phạt multiway pot.
/// - Dạy AI cách phòng thủ Preflop kỷ luật (4-Bet với bài tuyển, Fold bài bẫy).
#[derive(Clone)]
pub struct ThreeBetAggro;
impl Policy for ThreeBetAggro {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let edge = normalized_edge(eq, n);
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let chen = f[F_HAND_STRENGTH];
        let cat = f[F_HAND_CAT];
        let has_draw = f[F_FLUSH_DRAW] > 0.5 || f[F_STRAIGHT_DRAW] > 0.5;
        let is_preflop = f[F_STREET_PREFLOP] > 0.5;
        let is_late = f[F_POS_BTN] > 0.5 || f[F_POS_CO] > 0.5 || f[F_POS_SB] > 0.5;
        let rand = pseudo_rand(f);

        // Preflop:
        if is_preflop {
            if f[F_PUSHFOLD] > 0.5 {
                if chen >= 0.22 || edge > 1.30 {
                    return pick(m, &[Action::AllIn, Action::CheckCall]);
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }

            if facing {
                // Đang có đối thủ Open-Raise:
                // Nếu bài có tiềm năng (Pairs, Broadways, Suited: chen >= 0.24):
                // Tung ngay cú 3-Bet to (BetPot75) với 70% tần suất!
                if chen >= 0.24 && rand < 0.70 {
                    return pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall]);
                }
                if chen >= 0.40 {
                    // Monster: 4-Bet / Shove thẳng
                    return pick(m, &[Action::AllIn, Action::BetPot75, Action::CheckCall]);
                }
                if eq >= pot_odds && is_late {
                    return pick(m, &[Action::CheckCall]);
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            } else {
                // Chưa ai mở pot: Open-raise mạnh tay 75% pot
                let open_chen = if is_late { 0.22 } else { 0.28 };
                if chen >= open_chen || edge > 1.25 {
                    return pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall]);
                } else if f[F_TO_CALL_NORM] > 0.05 {
                    return pick(m, &[Action::Fold, Action::CheckCall]);
                } else {
                    return pick(m, &[Action::CheckCall]);
                }
            }
        }

        // Postflop:
        // C-bet liên tục 75% pot để duy trì áp lực agresison
        if !facing {
            if edge > 1.40 || has_draw || rand < 0.60 {
                pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
            } else {
                pick(m, &[Action::CheckCall])
            }
        } else {
            // Đối mặt cược: Raise tiếp nếu có bài (cat >= 1/9 hoặc draw)
            if cat >= 0.11 || has_draw {
                pick(m, &[Action::BetPot75, Action::CheckCall])
            } else if eq > pot_odds + 0.05 {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 7. BountyHunter (Kẻ Săn Đầu Người / Short-Stack Isolator):
/// - Chuyên săn lùng và cô lập các đối thủ short-stack (<= 12 BB).
/// - Dùng cú Re-shove All-In để đuổi người chơi khác ra, ép short-stack phải đánh 1-vs-1.
/// - Dạy AI Deep CFR cách sống còn khi rơi vào cảnh Short-stack và cách trảm đối thủ khi là Big-stack.
#[derive(Clone)]
pub struct BountyHunter;
impl Policy for BountyHunter {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let edge = normalized_edge(eq, n);
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let chen = f[F_HAND_STRENGTH];
        let cat = f[F_HAND_CAT];
        let is_preflop = f[F_STREET_PREFLOP] > 0.5;
        let is_short_opp = f[F_EFF_BB] <= 0.24 || f[F_PUSHFOLD] > 0.5; // Đối thủ ngắn chip <= 12 BB
        let my_stack_rich = f[F_STACK_RANK] >= 0.60;

        // Preflop:
        if is_preflop {
            if f[F_PUSHFOLD] > 0.5 {
                // Bản thân là Short-stack: Push chuẩn Nash
                if chen >= 0.22 || edge > 1.40 {
                    return pick(m, &[Action::AllIn, Action::CheckCall]);
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }

            // Nếu phát hiện có Short-stack tham gia pot và mình có nhiều chip:
            // BountyHunter tung cú Re-shove All-In hoặc BetPot75 để ISOLATE (Cô lập)!
            if is_short_opp && my_stack_rich {
                if chen >= 0.24 || edge > 1.25 {
                    return pick(m, &[Action::AllIn, Action::BetPot75, Action::CheckCall]);
                }
            }

            if facing {
                if chen >= 0.35 || edge > 1.70 {
                    return pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall]);
                } else if eq >= pot_odds {
                    return pick(m, &[Action::CheckCall]);
                }
                return pick(m, &[Action::Fold, Action::CheckCall]);
            } else {
                if chen >= 0.28 || edge > 1.35 {
                    return pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall]);
                } else if f[F_TO_CALL_NORM] > 0.05 {
                    return pick(m, &[Action::Fold, Action::CheckCall]);
                } else {
                    return pick(m, &[Action::CheckCall]);
                }
            }
        }

        // Postflop:
        // Đánh dồn dập ép short-stack phải all-in
        if is_short_opp {
            if cat >= 0.11 || edge > 1.20 {
                return pick(m, &[Action::AllIn, Action::BetPot75, Action::CheckCall]);
            }
        }

        if facing {
            if edge > 1.90 || cat >= 0.22 {
                pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall])
            } else if eq > pot_odds + 0.02 || edge > 1.50 {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        } else if edge > 1.50 || cat >= 0.11 {
            pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
        } else if edge > 1.15 {
            pick(m, &[Action::BetPot33, Action::CheckCall])
        } else {
            pick(m, &[Action::CheckCall])
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

pub fn baseline_opponents() -> Vec<(&'static str, Box<dyn Policy>)> {
    vec![
        ("AlwaysFold", Box::new(AlwaysFold)),
        ("Random", Box::new(UniformRandomPolicy)),
        ("CallStation", Box::new(CallStation)),
        ("Maniac", Box::new(Maniac)),
        ("TAG", Box::new(TightAggressive)),
        ("LagShark", Box::new(LagShark)),
        ("NashPush", Box::new(NashPushBot)),
        ("IcmSniper", Box::new(IcmSniper)),
        ("NitTrap", Box::new(NitTrap)),
        ("StickyFish", Box::new(StickyFish)),
        ("ThreeBetAggro", Box::new(ThreeBetAggro)),
        ("BountyHunter", Box::new(BountyHunter)),
    ]
}

// ===================================================================== Mô phỏng

#[derive(Default, Clone)]
struct Acc {
    n: u64,
    sum: f64,
    sumsq: f64,
    acts: [u64; NUM_ACTIONS],
    trash: u64,
    trash_shove: u64,
}

impl Acc {
    fn merge(mut self, o: &Acc) -> Acc {
        self.n += o.n;
        self.sum += o.sum;
        self.sumsq += o.sumsq;
        for i in 0..NUM_ACTIONS {
            self.acts[i] += o.acts[i];
        }
        self.trash += o.trash;
        self.trash_shove += o.trash_shove;
        self
    }
}

fn play_hand_table<R: Rng>(
    hero: &dyn Policy,
    opp: &dyn Policy,
    stack_bb: u32,
    acc: &mut Acc,
    rng: &mut R,
    num_players: usize,
) {
    let bb = 100u32;
    let sb = 50u32;
    let n = num_players.clamp(2, TABLE_SIZE);
    let stacks = vec![stack_bb * bb; n];
    let button = rng.gen_range(0..n);
    let hero_seat = rng.gen_range(0..n);

    let mut env = Environment::new(n, &stacks, bb, sb, button);
    env.reset_hand(rng, button);

    let mut guard = 0;
    while env.state.street != Street::Showdown && guard < 400 {
        guard += 1;
        let cur = env.state.current_player_idx;
        let mask = env.get_action_mask();
        if mask.0 == 0 {
            break;
        }
        let feats = env.state.encode_features(cur);
        let action = if cur == hero_seat {
            let a = hero.select_action(&feats, &mask);
            acc.acts[a.to_index()] += 1;
            let in_hand_cnt = env.state.players.iter().filter(|p| p.is_in_hand()).count();
            if feats[F_STREET_PREFLOP] > 0.5 && feats[F_HAND_STRENGTH] < trash_chen_threshold(in_hand_cnt) {
                acc.trash += 1;
                if a == Action::AllIn {
                    acc.trash_shove += 1;
                }
            }
            a
        } else {
            opp.select_action(&feats, &mask)
        };
        match env.step(action) {
            Ok(true) | Err(_) => break,
            Ok(false) => {}
        }
    }

    let end = env.state.players[hero_seat].stack as f64;
    let r = (end - (stack_bb * bb) as f64) / bb as f64;
    acc.n += 1;
    acc.sum += r;
    acc.sumsq += r * r;
}

#[allow(dead_code)]
fn play_hand<R: Rng>(
    hero: &dyn Policy,
    opp: &dyn Policy,
    stack_bb: u32,
    acc: &mut Acc,
    rng: &mut R,
) {
    play_hand_table(hero, opp, stack_bb, acc, rng, TABLE_SIZE);
}

fn run_match_table(
    hero: &dyn Policy,
    opp: &dyn Policy,
    hands: usize,
    stack_bb: u32,
    num_players: usize,
) -> Acc {
    (0..hands)
        .into_par_iter()
        .fold(Acc::default, |mut acc, _| {
            let mut rng = thread_rng();
            play_hand_table(hero, opp, stack_bb, &mut acc, &mut rng, num_players);
            acc
        })
        .reduce(Acc::default, |a, b| a.merge(&b))
}

#[allow(dead_code)]
fn run_match(hero: &dyn Policy, opp: &dyn Policy, hands: usize, stack_bb: u32) -> Acc {
    (0..hands)
        .into_par_iter()
        .fold(Acc::default, |mut acc, _| {
            let mut rng = thread_rng();
            play_hand(hero, opp, stack_bb, &mut acc, &mut rng);
            acc
        })
        .reduce(Acc::default, |a, b| a.merge(&b))
}

/// Chạy 1 ván trên bàn HỖN HỢP: Hero thi đấu cùng các đối thủ đa dạng:
/// 1 IcmSniper, 1 TAG, 1 LagShark, 1 NashPushBot, 1 CallStation, 1 Maniac, 1 Random (Mô phỏng chân thực bàn thi đấu tournament)
fn play_mixed_hand<R: Rng>(
    hero: &dyn Policy,
    stack_bb: u32,
    acc: &mut Acc,
    rng: &mut R,
) {
    let bb = 100u32;
    let sb = 50u32;
    let n = TABLE_SIZE;
    let stacks = vec![stack_bb * bb; n];
    let button = rng.gen_range(0..n);
    let hero_seat = rng.gen_range(0..n);

    let mut env = Environment::new(n, &stacks, bb, sb, button);
    env.reset_hand(rng, button);

    let offset = rng.gen_range(0..8);

    let mut guard = 0;
    while env.state.street != Street::Showdown && guard < 400 {
        guard += 1;
        let cur = env.state.current_player_idx;
        let mask = env.get_action_mask();
        if mask.0 == 0 {
            break;
        }
        let feats = env.state.encode_features(cur);
        let action = if cur == hero_seat {
            let a = hero.select_action(&feats, &mask);
            acc.acts[a.to_index()] += 1;
            let in_hand_cnt = env.state.players.iter().filter(|p| p.is_in_hand()).count();
            if feats[F_STREET_PREFLOP] > 0.5 && feats[F_HAND_STRENGTH] < trash_chen_threshold(in_hand_cnt) {
                acc.trash += 1;
                if a == Action::AllIn {
                    acc.trash_shove += 1;
                }
            }
            a
        } else {
            let opp_slot = (if cur > hero_seat { cur - 1 } else { cur } + offset) % 8;
            match opp_slot {
                0 => IcmSniper.select_action(&feats, &mask),
                1 => TightAggressive.select_action(&feats, &mask),
                2 => LagShark.select_action(&feats, &mask),
                3 => NashPushBot.select_action(&feats, &mask),
                4 => NitTrap.select_action(&feats, &mask),
                5 => StickyFish.select_action(&feats, &mask),
                6 => ThreeBetAggro.select_action(&feats, &mask),
                _ => BountyHunter.select_action(&feats, &mask),
            }
        };
        match env.step(action) {
            Ok(true) | Err(_) => break,
            Ok(false) => {}
        }
    }

    let end = env.state.players[hero_seat].stack as f64;
    let r = (end - (stack_bb * bb) as f64) / bb as f64;
    acc.n += 1;
    acc.sum += r;
    acc.sumsq += r * r;
}

fn run_mixed_match(hero: &dyn Policy, hands: usize, stack_bb: u32) -> Acc {
    (0..hands)
        .into_par_iter()
        .fold(Acc::default, |mut acc, _| {
            let mut rng = thread_rng();
            play_mixed_hand(hero, stack_bb, &mut acc, &mut rng);
            acc
        })
        .reduce(Acc::default, |a, b| a.merge(&b))
}

// ===================================================================== Kết quả

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub opponent: String,
    pub stack_bb: u32,
    pub hands: u64,
    pub bb_per_100: f64,
    pub ci95: f64,
    pub fold_pct: f64,
    pub shove_pct: f64,
    pub trash_shove_pct: f64,
}

fn to_result(name: &str, stack_bb: u32, a: &Acc) -> BenchResult {
    let n = a.n.max(1) as f64;
    let mean = a.sum / n;
    let var = (a.sumsq / n - mean * mean).max(0.0);
    let se = (var / n).sqrt();
    let total: u64 = a.acts.iter().sum::<u64>().max(1);
    BenchResult {
        opponent: name.to_string(),
        stack_bb,
        hands: a.n,
        bb_per_100: mean * 100.0,
        ci95: 1.96 * se * 100.0,
        fold_pct: a.acts[Action::Fold.to_index()] as f64 * 100.0 / total as f64,
        shove_pct: a.acts[Action::AllIn.to_index()] as f64 * 100.0 / total as f64,
        trash_shove_pct: if a.trash > 0 {
            a.trash_shove as f64 * 100.0 / a.trash as f64
        } else {
            0.0
        },
    }
}

/// Chạy cả bộ: từng đối thủ baseline đơn lẻ + bàn Mixed hỗn hợp
pub fn run_suite(hero: &dyn Policy, hands: usize, depths: &[u32]) -> Vec<BenchResult> {
    let mut out = Vec::new();
    for (name, opp) in baseline_opponents() {
        // Với đối thủ Random và Maniac: Đánh Heads-Up (2 người) để đo lường chính xác
        // khả năng trừng phạt lối chơi bừa bãi mà không bị hiệu ứng xổ số/bingo 8 người làm méo mó phương sai
        let num_players = if name == "Random" || name == "Maniac" {
            2
        } else {
            TABLE_SIZE
        };
        for &d in depths {
            let acc = run_match_table(hero, opp.as_ref(), hands, d, num_players);
            out.push(to_result(name, d, &acc));
        }
    }
    // Thêm bàn hỗn hợp MixedTable
    for &d in depths {
        let acc = run_mixed_match(hero, hands, d);
        out.push(to_result("MixedTable", d, &acc));
    }
    out
}

/// Hàm chấm điểm chuyên nghiệp cho Poker AI (Đánh giá đa tiêu chí & Trừng phạt lối chơi cờ bạc)
pub fn score(results: &[BenchResult]) -> f64 {
    let mut total_score = 0.0;
    let mut mixed_score = 0.0;
    let mut tag_score = 0.0;
    let mut max_trash_shove = 0.0f64;

    for r in results {
        // Thu thập tỷ lệ shove rác lớn nhất trên các bàn chuẩn mực (MixedTable, TAG, IcmSniper, ...)
        // (Không tính phạt khi khai thác bot đánh bừa như Random / Maniac)
        if r.opponent != "Random" && r.opponent != "Maniac" && r.trash_shove_pct > max_trash_shove {
            max_trash_shove = r.trash_shove_pct;
        }

        match r.opponent.as_str() {
            "MixedTable" => {
                // Bàn MixedTable là thước đo thực chiến quan trọng nhất
                mixed_score += r.bb_per_100.clamp(-200.0, 200.0);
            }
            "TAG" => {
                // TAG là thước đo độ hở sườn (Exploitability)
                tag_score += r.bb_per_100.clamp(-300.0, 100.0);
            }
            "IcmSniper" => {
                // Trùm cuối Heuristic: thước đo tiệm cận GTO cao cấp nhất
                total_score += r.bb_per_100.clamp(-200.0, 150.0) * 1.5;
            }
            "LagShark" => {
                total_score += r.bb_per_100.clamp(-200.0, 150.0) * 1.2;
            }
            "NashPush" => {
                total_score += r.bb_per_100.clamp(-200.0, 150.0) * 1.0;
            }
            "NitTrap" => {
                // Thước đo kỷ luật, kiểm soát pot và chống bị bẫy slow-play
                total_score += r.bb_per_100.clamp(-200.0, 150.0) * 1.2;
            }
            "StickyFish" => {
                // Thước đo khai thác calling station: value bet dày, không bluff vô ích
                // Giới hạn trần 50 để kiểm tra exploit mà không làm méo mó lạm phát điểm tổng
                total_score += r.bb_per_100.clamp(-50.0, 50.0) * 0.4;
            }
            "ThreeBetAggro" => {
                // Thước đo chống 3-bet bully và squeeze áp lực preflop
                total_score += r.bb_per_100.clamp(-200.0, 150.0) * 1.2;
            }
            "BountyHunter" => {
                // Thước đo đối phó áp lực short-stack và bubble hunter re-shove
                total_score += r.bb_per_100.clamp(-200.0, 150.0) * 1.1;
            }
            "CallStation" => {
                // Farm gà chỉ lấy tối đa 50 điểm để không làm méo mó thước đo
                total_score += r.bb_per_100.clamp(-50.0, 50.0) * 0.3;
            }
            "Maniac" => {
                // Heads-Up trừng phạt All-In bừa bãi
                total_score += r.bb_per_100.clamp(-50.0, 50.0) * 0.3;
            }
            "Random" => {
                // Bot ngẫu nhiên chỉ là bài test sàn tối thiểu, không làm rung lắc bảng điểm
                total_score += r.bb_per_100.clamp(-30.0, 30.0) * 0.1;
            }
            _ => {}
        }
    }

    // 1. Điểm cơ sở: Ưu tiên số 1 là MixedTable, ưu tiên số 2 là không thua TAG
    let mut final_score = (mixed_score * 1.5) + (tag_score * 1.2) + total_score;

    // 2. HÌNH PHẠT KỶ LUẬT (DISCIPLINE PENALTY):
    // Một AI Poker đỉnh cao không bao giờ được phép Shove bài rác quá 2%
    if max_trash_shove > 2.0 {
        let penalty = (max_trash_shove - 2.0) * 15.0; // Mỗi 1% shove rác phạt 15 điểm
        final_score -= penalty;
    }

    // 3. Nếu thua TAG quá sâu (chứng tỏ bot đang chơi mù quáng/hở sườn nặng) -> Trừ thêm điểm phạt
    if tag_score < -100.0 {
        final_score -= 50.0;
    }

    final_score
}

pub fn print_table(label: &str, results: &[BenchResult]) {
    println!("   ┌─ Benchmark: {} (score {:+.1} bb/100)", label, score(results));
    println!("   │ {:<12} {:>5} {:>8} {:>10} {:>8} {:>7} {:>7} {:>11}",
        "Opponent", "BB", "Hands", "bb/100", "±95%", "Fold%", "Shove%", "TrashShove%");
    for r in results {
        println!("   │ {:<12} {:>5} {:>8} {:>+10.1} {:>8.1} {:>7.1} {:>7.1} {:>11.1}",
            r.opponent, r.stack_bb, r.hands, r.bb_per_100, r.ci95,
            r.fold_pct, r.shove_pct, r.trash_shove_pct);
    }
    println!("   └─");
}

pub fn append_csv(path: &str, iter: usize, label: &str, results: &[BenchResult]) -> std::io::Result<()> {
    let new = !std::path::Path::new(path).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if new {
        writeln!(f, "iter,policy,opponent,stack_bb,hands,bb_per_100,ci95,fold_pct,shove_pct,trash_shove_pct")?;
    }
    for r in results {
        writeln!(
            f,
            "{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2}",
            iter, label, r.opponent, r.stack_bb, r.hands, r.bb_per_100, r.ci95,
            r.fold_pct, r.shove_pct, r.trash_shove_pct
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::state::FEATURE_DIM;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_feature_indices() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut env = Environment::new(6, &[10_000; 6], 100, 50, 0);
        env.reset_hand(&mut rng, 0);
        let cur = env.state.current_player_idx;
        let f = env.state.encode_features(cur);
        assert_eq!(f.len(), FEATURE_DIM);
        assert_eq!(f[F_STREET_PREFLOP], 1.0, "street preflop one-hot");
        assert!((f[F_IN_HAND] * 8.0 - 6.0).abs() < 1e-4, "in_hand/8");
        assert!(f[F_TO_CALL_NORM] > 0.0, "UTG đối mặt BB");
        assert!(f[F_EQUITY] > 0.0 && f[F_EQUITY] < 1.0, "equity");
        let to_call = 100.0f32;
        let pot = 150.0f32;
        assert!((f[F_POT_ODDS] - to_call / (pot + to_call)).abs() < 1e-4, "pot odds");
        assert!(f[F_HAND_STRENGTH] >= 0.0 && f[F_HAND_STRENGTH] <= 1.0);
        assert!(f[F_FACING_RAISE] == 0.0 || f[F_FACING_RAISE] == 1.0);
        assert!(f[F_RELATIVE_BET_SIZE] >= 0.0);
    }

    #[test]
    fn test_tag_self_symmetry() {
        // TAG đấu với 5 TAG (6-max): Kết quả lý thuyết sau khi trừ blind đối xứng phải tiệm cận 0 bb/100
        let acc = run_match_table(&TightAggressive, &TightAggressive, 6000, 50, 6);
        let res = to_result("TAG_vs_TAG", 50, &acc);
        println!("TAG vs 5 TAG: {:.2} ± {:.2} bb/100", res.bb_per_100, res.ci95);
        assert!(res.bb_per_100.abs() <= res.ci95.max(30.0), "Môi trường hoặc thứ tự chia bị lệch vị trí!");
    }

    #[test]
    fn test_always_fold_benchmark() {
        // Fold 100% trong bàn 6-max: lý thuyết trừ blind là -25 bb/100,
        // nhưng thực tế khi ở BB được check miễn phí hoặc đối thủ fold tới BB (walk),
        // tỷ lệ lỗ ròng thực tế rơi vào khoảng -11.0 đến -15.0 bb/100.
        let acc = run_match(&AlwaysFold, &TightAggressive, 4000, 50);
        let res = to_result("AlwaysFold", 50, &acc);
        println!("AlwaysFold vs TAG: {:.2} ± {:.2} bb/100", res.bb_per_100, res.ci95);
        assert!(
            res.bb_per_100 < -9.0 && res.bb_per_100 > -21.0,
            "AlwaysFold score out of expected range: {:.2}",
            res.bb_per_100
        );
    }

    #[test]
    fn test_run_suite_smoke() {
        let results = run_suite(&TightAggressive, 30, &[50]);
        assert_eq!(results.len(), 13); // 12 baseline opponents + MixedTable
        let _ = score(&results);
    }

    #[test]
    fn test_new_heuristic_bots_smoke() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = Environment::new(8, &[2000; 8], 100, 50, 0);
        env.reset_hand(&mut rng, 0);
        let cur = env.state.current_player_idx;
        let mask = env.get_action_mask();
        let feats = env.state.encode_features(cur);

        let bots: [&dyn Policy; 8] = [
            &TightAggressive,
            &LagShark,
            &NashPushBot,
            &IcmSniper,
            &NitTrap,
            &StickyFish,
            &ThreeBetAggro,
            &BountyHunter,
        ];

        for bot in bots {
            let act = bot.select_action(&feats, &mask);
            assert!(mask.is_valid(act));
            let probs = bot.get_action_probs(&feats, &mask);
            assert_eq!(probs[act.to_index()], 1.0);
        }
    }

    #[test]
    fn test_arena_heuristic_team_smoke() {
        let res = evaluate_against_heuristic_team(&TightAggressive, 100, 35);
        assert_eq!(res.hands, 100);
    }

    #[test]
    fn test_heads_up_bots_call_shove_with_strong_hands() {
        // Giả lập ván Heads-Up (2 người) ở mức 10 BB (Push/Fold mode)
        let mut f = vec![0.0f32; FEATURE_DIM];
        f[F_STREET_PREFLOP] = 1.0;
        f[F_IN_HAND] = 2.0 / 8.0; // 2 người trong ván bài -> n = 2.0
        f[F_TO_CALL_NORM] = 0.50; // Đối thủ vừa shove All-In
        f[F_POT_ODDS] = 0.48;
        f[F_EFF_BB] = 10.0 / 50.0;
        f[F_PUSHFOLD] = 1.0; // eff_stack < 12 BB
        f[F_EQUITY] = 0.85; // Cầm Pocket Aces (AA) ở Heads-Up
        f[F_HAND_STRENGTH] = 0.58; // Chen score của Pocket Aces (AA)

        let mut mask = ActionMask::EMPTY;
        mask.enable(Action::Fold);
        mask.enable(Action::CheckCall);
        mask.enable(Action::AllIn);

        // Cả 8 bot heuristic phải CALL hoặc ALL-IN, TUYỆT ĐỐI KHÔNG ĐƯỢC FOLD POCKET ACES!
        let bots: [(&str, &dyn Policy); 8] = [
            ("TAG", &TightAggressive),
            ("LagShark", &LagShark),
            ("NashPush", &NashPushBot),
            ("IcmSniper", &IcmSniper),
            ("NitTrap", &NitTrap),
            ("StickyFish", &StickyFish),
            ("ThreeBetAggro", &ThreeBetAggro),
            ("BountyHunter", &BountyHunter),
        ];

        for (name, bot) in bots {
            let act = bot.select_action(&f, &mask);
            assert_ne!(act, Action::Fold, "Bot {} không được fold Pocket Aces ở Heads-Up!", name);
        }
    }

    #[test]
    fn test_nashpush_v2_positional_shove() {
        // Cùng 1 tay bài biên (Chen = 0.18) ở mức stack 10 BB:
        let mut f_sb = vec![0.0f32; FEATURE_DIM];
        f_sb[F_STREET_PREFLOP] = 1.0;
        f_sb[F_IN_HAND] = 6.0 / 8.0;
        f_sb[F_EFF_BB] = 10.0 / 50.0;
        f_sb[F_HAND_STRENGTH] = 0.18; // K7o / Q8s
        f_sb[F_POS_SB] = 1.0; // Ở SB vs BB

        let mut mask = ActionMask::EMPTY;
        mask.enable(Action::Fold);
        mask.enable(Action::CheckCall);
        mask.enable(Action::AllIn);

        let sb_act = NashPushBot.select_action(&f_sb, &mask);
        assert_eq!(sb_act, Action::AllIn, "NashPushBot ở SB vs BB 10 BB phải Open-Shove!");

        // Nhưng ở UTG (Early Position) thì phải Fold bài Chen 0.18!
        let mut f_utg = f_sb.clone();
        f_utg[F_POS_SB] = 0.0;
        f_utg[F_POS_BTN] = 0.0;
        f_utg[F_POS_CO] = 0.0;
        let utg_act = NashPushBot.select_action(&f_utg, &mask);
        assert_eq!(utg_act, Action::Fold, "NashPushBot ở UTG 10 BB phải Fold bài yếu (Chen 0.18)!");
    }

    #[test]
    fn test_tag_v2_wet_board_protection() {
        let mut f = vec![0.0f32; FEATURE_DIM];
        f[F_STREET_PREFLOP] = 0.0;
        f[F_STREET_FLOP] = 1.0;
        f[F_IN_HAND] = 3.0 / 8.0; // 3 người
        f[F_HAND_CAT] = 0.11;     // Top Pair
        f[F_EQUITY] = 0.65;
        f[F_MAX_SUIT_BOARD] = 0.60; // 3 lá cùng chất trên bàn (Wet Board)

        let mut mask = ActionMask::EMPTY;
        mask.enable(Action::CheckCall);
        mask.enable(Action::BetSmall);
        mask.enable(Action::BetPot33);
        mask.enable(Action::BetPot75);

        let act = TightAggressive.select_action(&f, &mask);
        assert_eq!(act, Action::BetPot75, "TAG v2 phải Bet 75% để bảo vệ Top Pair trên mặt bài ướt!");
    }

    #[test]
    fn test_icmsniper_v2_bubble_chip_leader_pressure() {
        let mut f = vec![0.0f32; FEATURE_DIM];
        f[F_IS_BUBBLE] = 1.0;  // Vòng Bubble
        f[F_STACK_RANK] = 0.90; // Chip Leader
        f[F_IN_HAND] = 5.0 / 8.0;

        let mut mask = ActionMask::EMPTY;
        mask.enable(Action::CheckCall);
        mask.enable(Action::BetPot33);
        mask.enable(Action::BetPot75);

        let act = IcmSniper.select_action(&f, &mask);
        assert_eq!(act, Action::BetPot75, "IcmSniper làm Chip Leader ở Bubble phải BetPot75 ép bàn!");
    }

    #[test]
    fn test_four_new_heuristic_archetypes() {
        // 1. NitTrap: Preflop đối mặt với Bet cầm bài yếu (Chen 0.20) -> Phải FOLD ngay lập tức!
        let mut f_nit = vec![0.0f32; FEATURE_DIM];
        f_nit[F_STREET_PREFLOP] = 1.0;
        f_nit[F_TO_CALL_NORM] = 0.15; // Có người raise 3 BB
        f_nit[F_HAND_STRENGTH] = 0.20; // Bài yếu (Chen 0.20)
        let mut mask = ActionMask::EMPTY;
        mask.enable(Action::Fold);
        mask.enable(Action::CheckCall);
        mask.enable(Action::BetPot75);
        let nit_act = NitTrap.select_action(&f_nit, &mask);
        assert_eq!(nit_act, Action::Fold, "NitTrap cầm bài yếu đối mặt với Bet phải Fold!");

        // 2. StickyFish: Flop đối mặt với Bet 50% pot khi cầm One Pair -> Dứt khoát CALL (không bao giờ Fold)!
        let mut f_fish = vec![0.0f32; FEATURE_DIM];
        f_fish[F_STREET_FLOP] = 1.0;
        f_fish[F_TO_CALL_NORM] = 0.10;
        f_fish[F_POT_ODDS] = 0.25;
        f_fish[F_HAND_CAT] = 0.11; // One Pair
        f_fish[F_EQUITY] = 0.40;
        let mut fish_mask = ActionMask::EMPTY;
        fish_mask.enable(Action::Fold);
        fish_mask.enable(Action::CheckCall);
        fish_mask.enable(Action::BetPot75);
        let fish_act = StickyFish.select_action(&f_fish, &fish_mask);
        assert_eq!(fish_act, Action::CheckCall, "StickyFish cầm One Pair đối mặt Bet vừa phải tuyệt đối không Fold!");

        // StickyFish v2: Đối mặt All-In cực to khi chỉ có bài yếu (eq thấp, chen thấp) -> Biết Fold tránh tự sát!
        let mut f_fish_huge = f_fish.clone();
        f_fish_huge[F_TO_CALL_RATIO_STACK] = 0.80; // Shove 80% stack
        f_fish_huge[F_TO_CALL_NORM] = 0.40;
        f_fish_huge[F_EQUITY] = 0.25;
        f_fish_huge[F_HAND_STRENGTH] = 0.15;
        let fish_huge_act = StickyFish.select_action(&f_fish_huge, &fish_mask);
        assert_eq!(fish_huge_act, Action::Fold, "StickyFish gặp cú Shove All-In cực to với bài yếu phải biết Fold!");

        // StickyFish v2: Cầm Two Pair (cat 0.22) khi chưa ai bet -> Phải Bet lấy tiền, không check thụ động!
        let mut f_fish_twopair = f_fish.clone();
        f_fish_twopair[F_TO_CALL_NORM] = 0.0;
        f_fish_twopair[F_HAND_CAT] = 0.22;
        let mut fish_bet_mask = ActionMask::EMPTY;
        fish_bet_mask.enable(Action::CheckCall);
        fish_bet_mask.enable(Action::BetPot33);
        fish_bet_mask.enable(Action::BetPot75);
        let fish_tp_act = StickyFish.select_action(&f_fish_twopair, &fish_bet_mask);
        assert!(
            fish_tp_act == Action::BetPot75 || fish_tp_act == Action::BetPot33,
            "StickyFish cầm Two Pair phải chủ động Bet lấy value!"
        );

        // 3. ThreeBetAggro: Preflop đối mặt với Open-Raise cầm bài khủng (Chen 0.48) -> Phải 3-Bet Pot75 hoặc AllIn!
        let mut f_aggro = vec![0.0f32; FEATURE_DIM];
        f_aggro[F_STREET_PREFLOP] = 1.0;
        f_aggro[F_TO_CALL_NORM] = 0.15;
        f_aggro[F_HAND_STRENGTH] = 0.48; // AK, QQ
        f_aggro[F_IN_HAND] = 6.0 / 8.0;
        let mut aggro_mask = ActionMask::EMPTY;
        aggro_mask.enable(Action::Fold);
        aggro_mask.enable(Action::CheckCall);
        aggro_mask.enable(Action::BetPot75);
        aggro_mask.enable(Action::AllIn);
        let aggro_act = ThreeBetAggro.select_action(&f_aggro, &aggro_mask);
        assert!(
            aggro_act == Action::BetPot75 || aggro_act == Action::AllIn,
            "ThreeBetAggro cầm AK/QQ đối mặt Open-Raise phải 3-Bet tấn công!"
        );

        // 4. BountyHunter: Mình nhiều chip (Stack rank 0.80), đối thủ Short Stack (8 BB) -> Đẩy All-In săn đầu người ngay!
        let mut f_bounty = vec![0.0f32; FEATURE_DIM];
        f_bounty[F_STREET_PREFLOP] = 1.0;
        f_bounty[F_STACK_RANK] = 0.80; // Nhiều chip hơn đối thủ
        f_bounty[F_EFF_BB] = 8.0 / 50.0; // Short stack 8 BB
        f_bounty[F_HAND_STRENGTH] = 0.35; // JJ, AJs
        f_bounty[F_IN_HAND] = 3.0 / 8.0;
        let mut bounty_mask = ActionMask::EMPTY;
        bounty_mask.enable(Action::Fold);
        bounty_mask.enable(Action::CheckCall);
        bounty_mask.enable(Action::BetPot75);
        bounty_mask.enable(Action::AllIn);
        let bounty_act = BountyHunter.select_action(&f_bounty, &bounty_mask);
        assert_eq!(bounty_act, Action::AllIn, "BountyHunter thấy Short Stack 8 BB với bài đẹp phải Shove All-In!");
    }
}

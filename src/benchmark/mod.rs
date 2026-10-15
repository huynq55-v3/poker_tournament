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
    evaluate_against_champion, evaluate_against_heuristic_team, evaluate_arena_suite,
    print_arena_results, ArenaMatchResult,
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
const F_STREET_RIVER: usize = 126;
const F_TO_CALL_NORM: usize = 128; // to_call / (20 * BB) -> 1 BB = 0.05
const F_POT_ODDS: usize = 130;
const F_IN_HAND: usize = 160;
const F_EFF_BB: usize = 161;        // eff_stack_bb / 50.0
const F_PUSHFOLD: usize = 162;
const F_FACING_BET: usize = 163;
const F_HAND_STRENGTH: usize = 167; // preflop: điểm Chen 0..1
const F_EQUITY: usize = 168;
const F_AGG_BARRELS: usize = 170;   // Số vòng đối thủ đã barrel / 4.0
const F_FLUSH_DRAW: usize = 171;
const F_STRAIGHT_DRAW: usize = 172;
#[allow(dead_code)]
const F_BOARD_HIGH: usize = 175;

/// Ở Preflop, chỉ coi là đang đối mặt với Bet/Raise nếu to_call > 1 BB (đã có người Open-Raise).
/// Nếu to_call <= 1 BB thì là Unopened Pot -> Quân xanh phải chủ động Open-Raise bài mạnh hoặc Fold bài rác!
#[inline(always)]
fn is_facing_real_bet(f: &[f32]) -> bool {
    if f[F_STREET_PREFLOP] > 0.5 {
        f[F_TO_CALL_NORM] > 0.055
    } else {
        f[F_FACING_BET] > 0.5
    }
}

/// Bài rác preflop thực sự (72o, 83o, 93o, J2o...): điểm Chen chuẩn hóa dưới 0.22
/// (Không tính nhầm các bài 22..55, A9o, KTo, QJs đang push đúng Nash ở 12BB là rác!)
const TRASH_CHEN: f32 = 0.22;
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

/// Chặt-hung hăng dựa trên equity, vị trí và khả năng bắt bluff All-in chuẩn xác
#[derive(Clone)]
pub struct TightAggressive;
impl Policy for TightAggressive {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let edge = eq * n; // 1.0 = bài trung bình trên bàn
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let chen = f[F_HAND_STRENGTH]; // Điểm Chen ở Preflop (0..1)

        // 1. Chế độ Push/Fold (< 12 BB):
        if f[F_PUSHFOLD] > 0.5 {
            // Không đẩy bài rác, chỉ đẩy/call khi bài thuộc top (edge > 1.65 hoặc Chen >= 0.45)
            if edge > 1.75 || (f[F_STREET_PREFLOP] > 0.5 && chen >= 0.48) {
                return pick(m, &[Action::AllIn, Action::CheckCall]);
            } else if facing {
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }
        }

        // 2. Khi đối mặt với Bet / Raise / All-In của đối thủ:
        if facing {
            // 🟢 1. BẮT CHẾT KẺ RE-SHOVE LÁO Ở PREFLOP:
            // Hạ ngưỡng từ 0.52 xuống 0.42 (bao gồm 88+, ATs+, KTs+, AJo+, KQo - Top 11% bài):
            // Hễ đã cầm bài tốt ở Preflop là KHÔNG BAO GIỜ FOLD trước cú All-in cướp pot!
            if f[F_STREET_PREFLOP] > 0.5 && (chen >= 0.42 || edge > 1.50) {
                if chen >= 0.60 || edge > 2.0 {
                    return pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall]);
                } else {
                    return pick(m, &[Action::CheckCall]);
                }
            }

            // 🟢 2. BẮT CHẾT KẺ SHOVE LÁO Ở POSTFLOP (Flop/Turn/River):
            // Nếu có bài khá mạnh ở Postflop (edge > 1.55, ví dụ Top Pair+), luôn Call dù bị đối thủ Shove!
            if edge > 2.1 {
                pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
            } else if eq > pot_odds + 0.03 || edge > 1.55 {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        } else if edge > 1.85 {
            pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
        } else if edge > 1.35 {
            pick(m, &[Action::BetPot33, Action::CheckCall])
        } else if edge > 1.10 && f[F_STREET_PREFLOP] < 0.5 {
            // C-bet nhỏ / Block bet thăm dò ở Postflop
            pick(m, &[Action::BetSmall, Action::CheckCall])
        } else if f[F_STREET_PREFLOP] > 0.5 && f[F_FACING_BET] > 0.5 {
            // Ở Preflop khi bài yếu không đủ chuẩn Open-Raise: Fold luôn thay vì Limp nộp mạng!
            pick(m, &[Action::Fold, Action::CheckCall])
        } else {
            pick(m, &[Action::CheckCall])
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 1. LAG Shark (Loose-Aggressive):
/// - Tận dụng vị trí cuối (BTN, CO) để mở rộng range tấn công.
/// - Biết Semi-bluff (BetPot75/33) khi có Flush Draw hoặc Straight Draw ở Flop/Turn.
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

        // Cộng thêm Equity kỳ vọng khi có thùng/sảnh chờ hoặc lợi thế vị trí
        let draw_bonus = if has_draw && f[F_STREET_RIVER] < 0.5 { 0.12 } else { 0.0 };
        let pos_bonus = if is_late_pos && !facing { 0.08 } else { 0.0 };
        let eff_eq = (eq + draw_bonus + pos_bonus).min(0.98);
        let edge = eff_eq * n;

        if f[F_PUSHFOLD] > 0.5 && edge > 2.1 {
            return pick(m, &[Action::AllIn, Action::CheckCall]);
        }

        if facing {
            // Snap-call/raise với bài tốt ở Preflop (chen >= 0.42 || edge > 1.50) để chống trash shove
            if f[F_STREET_PREFLOP] > 0.5 && (chen >= 0.42 || edge > 1.50) {
                return pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall]);
            }

            if edge > 2.1 || (has_draw && eq > pot_odds) {
                // Semi-bluff raise hoặc Value raise
                pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
            } else if eff_eq > pot_odds + 0.02 || edge > 1.55 {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        } else if edge > 1.75 || (has_draw && edge > 1.25) {
            pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
        } else if edge > 1.30 || (is_late_pos && f[F_STREET_PREFLOP] > 0.5 && edge > 0.95) {
            // Steal blind ở vị trí cuối hoặc C-bet 33% chuẩn
            pick(m, &[Action::BetPot33, Action::BetSmall, Action::CheckCall])
        } else if edge > 1.05 && f[F_STREET_PREFLOP] < 0.5 {
            // Probe-bet / Block-bet nhỏ 15% ở Postflop với bài mỏng
            pick(m, &[Action::BetSmall, Action::CheckCall])
        } else if f[F_STREET_PREFLOP] > 0.5 && f[F_FACING_BET] > 0.5 {
            pick(m, &[Action::Fold, Action::CheckCall])
        } else {
            pick(m, &[Action::CheckCall])
        }
    }
    fn get_action_probs(&self, f: &[f32], m: &ActionMask) -> [f32; NUM_ACTIONS] {
        one_hot(self.select_action(f, m))
    }
}

/// 2. NashPushBot (Chuyên gia Short-Stack & Push/Fold):
/// - Khi stack <= 15 BB: Đánh chuẩn lý thuyết Nash Push/Fold theo số đối thủ còn lại và vị trí.
/// - Khi stack > 15 BB: Đánh kỷ luật chặt chẽ.
#[derive(Clone)]
pub struct NashPushBot;
impl Policy for NashPushBot {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let edge = eq * n;
        let pot_odds = f[F_POT_ODDS];
        let facing = is_facing_real_bet(f);
        let eff_bb = f[F_EFF_BB] * 50.0;
        let is_late = f[F_POS_BTN] > 0.5 || f[F_POS_SB] > 0.5 || f[F_POS_CO] > 0.5;
        let chen = f[F_HAND_STRENGTH];

        // Chế độ Short-stack Nash Push/Fold (<= 15 BB)
        if eff_bb <= 15.0 && f[F_STREET_PREFLOP] > 0.5 {
            if !facing {
                // Open-Shove ngưỡng linh hoạt theo độ sâu stack và vị trí
                let push_thresh = if is_late { 1.15 + (eff_bb / 45.0) } else { 1.45 + (eff_bb / 35.0) };
                if edge >= push_thresh {
                    return pick(m, &[Action::AllIn, Action::BetPot75, Action::CheckCall]);
                } else {
                    return pick(m, &[Action::Fold, Action::CheckCall]);
                }
            } else {
                // Call All-in cần ngưỡng chặt hơn Open-Shove (Gap Concept của Sklansky)
                // nhưng bắt chết với bài tốt (chen >= 0.42) tránh bị multiway eq làm mù quáng
                if (eq >= pot_odds + 0.06 || chen >= 0.42) && edge >= 1.45 {
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

/// 3. IcmSniper (TRÙM CUỐI HEURISTIC - Apex SNG Bot):
/// - Khấu trừ Equity khi đối thủ barrel nhiều vòng (không bao giờ làm CallStation nộp mạng).
/// - Áp dụng Bubble Factor (ép bàn khi làm Chip Leader, thu mình khi stack trung bình ở Bubble).
/// - Kết hợp cả Semi-bluff lẫn Nash Push/Fold.
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

        // 1. Đọc lịch sử barrel của đối thủ: Nếu đối thủ bet liên tục qua nhiều vòng,
        // IcmSniper tự hiểu đối thủ đang cầm top range và khấu trừ equity ảo!
        let opp_barrels = f[F_AGG_BARRELS] * 4.0; // 0..4 vòng
        let range_discount = if facing { (opp_barrels * 0.065).min(0.22) } else { 0.0 };

        // 2. Tính áp lực ICM / Bubble Factor:
        let is_bubble = f[F_IS_BUBBLE] > 0.5;
        let stack_rank = f[F_STACK_RANK]; // 1.0 = giàu nhất bàn
        let icm_tax = if is_bubble && facing && stack_rank < 0.85 {
            0.05 // Đang ở Bubble mà bị bet vào mặt -> đòi thêm 5% equity mới dám Call
        } else if stack_rank > 0.80 && !facing {
            -0.04 // Là Chip Leader và chưa ai bet -> giảm ngưỡng để gây áp lực lên bàn
        } else {
            0.0
        };

        let draw_bonus = if has_draw { 0.09 } else { 0.0 };
        let real_eq = (raw_eq + draw_bonus - range_discount).clamp(0.0, 1.0);
        let edge = real_eq * n;

        // Chế độ Push/Fold (< 12 BB) có điều chỉnh ICM
        if f[F_PUSHFOLD] > 0.5 {
            let push_req = if facing { 1.85 + icm_tax * 4.0 } else if is_late { 1.30 } else { 1.60 };
            if edge >= push_req && (!facing || real_eq >= pot_odds + 0.05 + icm_tax || chen >= 0.42) {
                return pick(m, &[Action::AllIn, Action::CheckCall]);
            } else {
                return pick(m, &[Action::Fold, Action::CheckCall]);
            }
        }

        if facing {
            // 🟢 BẮT CHẾT KẺ RE-SHOVE LÁO Ở PREFLOP:
            // Hạ ngưỡng xuống 0.42 (88+, ATs+, KTs+, AJo+, KQo - Top 11% bài)
            if f[F_STREET_PREFLOP] > 0.5 && (chen >= 0.42 || edge > 1.50) {
                if chen >= 0.60 || edge > 2.0 {
                    return pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall]);
                } else {
                    return pick(m, &[Action::CheckCall]);
                }
            }

            if edge > 2.35 {
                // Bài cực mạnh (Set, Two Pair lớn, Nut Flush) -> Raise lấy max value
                pick(m, &[Action::BetPot75, Action::AllIn, Action::CheckCall])
            } else if edge > 1.95 {
                pick(m, &[Action::BetPot75, Action::BetPot33, Action::CheckCall])
            } else if real_eq > pot_odds + 0.04 + icm_tax || edge > 1.60 {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        } else if edge > 2.0 {
            pick(m, &[Action::BetPot75, Action::CheckCall])
        } else if edge > 1.35 || (is_late && edge > 1.12) {
            // Ưu tiên BetPot33 trước, không để BetSmall đứng đầu làm yếu cú C-bet!
            pick(m, &[Action::BetPot33, Action::BetSmall, Action::CheckCall])
        } else if edge > 1.05 && f[F_STREET_PREFLOP] < 0.5 {
            // Block-bet nhỏ 15% ở Postflop
            pick(m, &[Action::BetSmall, Action::CheckCall])
        } else if f[F_STREET_PREFLOP] > 0.5 && f[F_FACING_BET] > 0.5 {
            pick(m, &[Action::Fold, Action::CheckCall])
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

fn play_hand<R: Rng>(
    hero: &dyn Policy,
    opp: &dyn Policy,
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
            if feats[F_STREET_PREFLOP] > 0.5 && feats[F_HAND_STRENGTH] < TRASH_CHEN {
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

    let opponents: [Box<dyn Policy>; 7] = [
        Box::new(IcmSniper),           // 1 Cao thủ SNG thực thụ
        Box::new(TightAggressive),     // 1 Người chơi chặt (TAG)
        Box::new(LagShark),            // 1 Cá mập đánh biến ảo (LAG)
        Box::new(NashPushBot),         // 1 Chuyên gia Push/Fold
        Box::new(CallStation),         // 1 Gà thích theo cược
        Box::new(Maniac),              // 1 Kẻ điên thích All-in
        Box::new(UniformRandomPolicy), // 1 Người chơi ngẫu nhiên
    ];

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
            if feats[F_STREET_PREFLOP] > 0.5 && feats[F_HAND_STRENGTH] < TRASH_CHEN {
                acc.trash += 1;
                if a == Action::AllIn {
                    acc.trash_shove += 1;
                }
            }
            a
        } else {
            let opp_slot = if cur > hero_seat { cur - 1 } else { cur } % 7;
            opponents[opp_slot].select_action(&feats, &mask)
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
        for &d in depths {
            let acc = run_match(hero, opp.as_ref(), hands, d);
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
        // Thu thập tỷ lệ shove rác lớn nhất trên toàn bộ các bàn
        if r.trash_shove_pct > max_trash_shove {
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
            "CallStation" => {
                // Farm gà chỉ lấy tối đa 150 điểm để không làm méo mó thước đo
                total_score += r.bb_per_100.clamp(-100.0, 150.0) * 0.5;
            }
            "Maniac" => {
                total_score += r.bb_per_100.clamp(-100.0, 100.0) * 0.8;
            }
            "Random" => {
                total_score += r.bb_per_100.clamp(-100.0, 100.0) * 0.3;
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
        assert_eq!(f[F_FACING_BET], 1.0, "UTG đối mặt BB");
        assert!(f[F_EQUITY] > 0.0 && f[F_EQUITY] < 1.0, "equity");
        let to_call = 100.0f32;
        let pot = 150.0f32;
        assert!((f[F_POT_ODDS] - to_call / (pot + to_call)).abs() < 1e-4, "pot odds");
        assert!(f[F_HAND_STRENGTH] >= 0.0 && f[F_HAND_STRENGTH] <= 1.0);
    }

    #[test]
    fn test_tag_self_symmetry() {
        // TAG đấu với 5 TAG: Kết quả lý thuyết sau khi trừ blind đối xứng phải tiệm cận 0 bb/100
        let acc = run_match(&TightAggressive, &TightAggressive, 4000, 50);
        let res = to_result("TAG_vs_TAG", 50, &acc);
        println!("TAG vs 5 TAG: {:.2} ± {:.2} bb/100", res.bb_per_100, res.ci95);
        assert!(res.bb_per_100.abs() <= res.ci95.max(20.0), "Môi trường hoặc thứ tự chia bị lệch vị trí!");
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
    fn test_new_heuristic_bots_smoke() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = Environment::new(8, &[2000; 8], 100, 50, 0);
        env.reset_hand(&mut rng, 0);
        let cur = env.state.current_player_idx;
        let mask = env.get_action_mask();
        let feats = env.state.encode_features(cur);

        let lag_action = LagShark.select_action(&feats, &mask);
        assert!(mask.is_valid(lag_action));
        let nash_action = NashPushBot.select_action(&feats, &mask);
        assert!(mask.is_valid(nash_action));
        let icm_action = IcmSniper.select_action(&feats, &mask);
        assert!(mask.is_valid(icm_action));

        let lag_probs = LagShark.get_action_probs(&feats, &mask);
        assert_eq!(lag_probs[lag_action.to_index()], 1.0);
        let nash_probs = NashPushBot.get_action_probs(&feats, &mask);
        assert_eq!(nash_probs[nash_action.to_index()], 1.0);
        let icm_probs = IcmSniper.get_action_probs(&feats, &mask);
        assert_eq!(icm_probs[icm_action.to_index()], 1.0);
    }

    #[test]
    fn test_arena_heuristic_team_smoke() {
        let res = evaluate_against_heuristic_team(&TightAggressive, 100, 35);
        assert_eq!(res.hands, 100);
    }
}

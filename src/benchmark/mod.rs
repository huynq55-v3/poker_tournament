//! Benchmark: đo sức mạnh policy bằng bb/100 so với các bot baseline và bàn hỗn hợp.

use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::self_play::{Policy, UniformRandomPolicy};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::io::Write;

pub mod arena;
pub use arena::{evaluate_against_champion, ArenaMatchResult};

// ---- Vị trí các đặc trưng trong encode_features (đổi layout thì phải cập nhật; có test kiểm tra) ----
const F_STREET_PREFLOP: usize = 119;
const F_POT_ODDS: usize = 126;
const F_IN_HAND: usize = 156;
const F_PUSHFOLD: usize = 158;
const F_FACING_BET: usize = 159;
const F_HAND_STRENGTH: usize = 163; // preflop: điểm Chen 0..1
const F_EQUITY: usize = 164;

/// Bài rác preflop: điểm Chen chuẩn hóa dưới ngưỡng này
const TRASH_CHEN: f32 = 0.30;
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

/// Chặt-hung hăng dựa trên equity so với bài ngẫu nhiên và pot odds
#[derive(Clone)]
pub struct TightAggressive;
impl Policy for TightAggressive {
    fn select_action(&self, f: &[f32], m: &ActionMask) -> Action {
        let eq = f[F_EQUITY];
        let n = (f[F_IN_HAND] * 8.0).round().max(2.0);
        let edge = eq * n; // 1.0 = phần chia công bằng
        let pot_odds = f[F_POT_ODDS];
        let facing = f[F_FACING_BET] > 0.5;

        if edge > 2.6 && f[F_PUSHFOLD] > 0.5 {
            return pick(m, &[Action::AllIn, Action::CheckCall]);
        }
        if facing {
            if edge > 2.2 {
                pick(m, &[Action::BetPot66, Action::BetPot33, Action::CheckCall])
            } else if eq > pot_odds + 0.04 {
                pick(m, &[Action::CheckCall])
            } else {
                pick(m, &[Action::Fold, Action::CheckCall])
            }
        } else if edge > 1.9 {
            pick(m, &[Action::BetPot66, Action::BetPot33, Action::CheckCall])
        } else if edge > 1.35 {
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
/// 1 TAG, 1 Maniac, 2 CallStation, 1 Random (Mô phỏng chân thực bàn thi đấu tournament)
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
        Box::new(TightAggressive),
        Box::new(CallStation),
        Box::new(Maniac),
        Box::new(CallStation),
        Box::new(UniformRandomPolicy),
        Box::new(TightAggressive),
        Box::new(UniformRandomPolicy),
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
        let mut env = Environment::new(6, &vec![10_000; 6], 100, 50, 0);
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
        assert!(res.bb_per_100.abs() < 15.0, "Môi trường hoặc thứ tự chia bị lệch vị trí!");
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
            res.bb_per_100 < -9.0 && res.bb_per_100 > -18.0,
            "AlwaysFold score out of expected range: {:.2}",
            res.bb_per_100
        );
    }
}

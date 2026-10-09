//! Benchmark: đo sức mạnh policy bằng bb/100 so với các bot baseline.

use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::self_play::{Policy, UniformRandomPolicy};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::io::Write;

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
const TABLE_SIZE: usize = 6;

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

/// Chạy cả bộ: mỗi đối thủ x mỗi độ sâu stack, `hands` ván cho mỗi ô.
pub fn run_suite(hero: &dyn Policy, hands: usize, depths: &[u32]) -> Vec<BenchResult> {
    let mut out = Vec::new();
    for (name, opp) in baseline_opponents() {
        for &d in depths {
            let acc = run_match(hero, opp.as_ref(), hands, d);
            out.push(to_result(name, d, &acc));
        }
    }
    out
}

/// Điểm tổng hợp = trung bình bb/100 trên mọi ô (dùng để chọn checkpoint tốt nhất)
/// Kẹp mỗi ô về [-300, 300] để một ô (CallStation) không chi phối;
/// TAG và Maniac tính gấp đôi vì là phép thử có ý nghĩa nhất.
pub fn score(results: &[BenchResult]) -> f64 {
    let (mut s, mut w) = (0.0, 0.0);
    for r in results {
        let weight = match r.opponent.as_str() {
            "TAG" | "Maniac" => 2.0,
            _ => 1.0,
        };
        s += weight * r.bb_per_100.clamp(-300.0, 300.0);
        w += weight;
    }
    if w > 0.0 { s / w } else { 0.0 }
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

    /// Kiểm tra hằng số chỉ số đặc trưng còn khớp với encode_features
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
    fn test_suite_runs() {
        let res = run_suite(&TightAggressive, 50, &[100]);
        assert_eq!(res.len(), 4);
        assert!(res.iter().all(|r| r.hands == 50));
    }

    /// Chạy tay: cargo test --release tag_beats_random -- --ignored --nocapture
    #[test]
    #[ignore]
    fn tag_beats_random() {
        let res = run_suite(&TightAggressive, 5000, &[100]);
        print_table("TAG", &res);
        let r = res.iter().find(|r| r.opponent == "Random").unwrap();
        assert!(r.bb_per_100 > 0.0);
    }
}

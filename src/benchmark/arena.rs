use crate::benchmark::{IcmSniper, LagShark, NashPushBot, TightAggressive};
use crate::environment::action::Action;
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::self_play::Policy;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

const F_STREET_PREFLOP: usize = 123;
const F_HAND_STRENGTH: usize = 167;
/// Bài rác preflop thực sự (72o, 83o, 93o, J2o...): điểm Chen chuẩn hóa dưới 0.22
const TRASH_CHEN: f32 = 0.22;
const ARENA_PLAYERS: usize = 8;

#[derive(Default, Clone)]
struct ArenaAcc {
    hands: u64,
    sum_diff: f64,
    sum_sq_diff: f64,
    challenger_shoves: u64,
    challenger_trash: u64,
    challenger_trash_shoves: u64,
    total_actions: u64,
}

impl ArenaAcc {
    fn merge(mut self, o: &ArenaAcc) -> ArenaAcc {
        self.hands += o.hands;
        self.sum_diff += o.sum_diff;
        self.sum_sq_diff += o.sum_sq_diff;
        self.challenger_shoves += o.challenger_shoves;
        self.challenger_trash += o.challenger_trash;
        self.challenger_trash_shoves += o.challenger_trash_shoves;
        self.total_actions += o.total_actions;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ArenaMatchResult {
    pub hands: u64,
    pub challenger_bb_per_100: f64,
    pub ci95: f64,
    pub challenger_trash_shove_pct: f64,
    pub is_winner: bool,
}

fn play_arena_team_hand<R: Rng>(
    challenger: &dyn Policy,
    opponents: &[&dyn Policy],
    stack_bb: u32,
    acc: &mut ArenaAcc,
    rng: &mut R,
    num_players: usize,
) {
    let bb = 100u32;
    let sb = 50u32;
    let n = num_players.clamp(2, 8);
    let stacks = vec![stack_bb * bb; n];
    let button = rng.gen_range(0..n);

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
        let is_challenger = cur.is_multiple_of(2);

        let action = if is_challenger {
            let a = challenger.select_action(&feats, &mask);
            acc.total_actions += 1;
            if a == Action::AllIn {
                acc.challenger_shoves += 1;
            }
            if feats[F_STREET_PREFLOP] > 0.5 && feats[F_HAND_STRENGTH] < TRASH_CHEN {
                acc.challenger_trash += 1;
                if a == Action::AllIn {
                    acc.challenger_trash_shoves += 1;
                }
            }
            a
        } else {
            let opp_idx = (cur / 2) % opponents.len();
            opponents[opp_idx].select_action(&feats, &mask)
        };

        match env.step(action) {
            Ok(true) | Err(_) => break,
            Ok(false) => {}
        }
    }

    let challenger_seats: Vec<usize> = (0..n).filter(|&seat| seat % 2 == 0).collect();
    let num_challengers = challenger_seats.len().max(1) as f64;
    let initial_team_chips = num_challengers * (stack_bb * bb) as f64;
    let final_team_chips: f64 = challenger_seats
        .iter()
        .map(|&seat| env.state.players[seat].stack as f64)
        .sum();

    let diff_bb = (final_team_chips - initial_team_chips) / (bb as f64) / num_challengers;

    acc.hands += 1;
    acc.sum_diff += diff_bb;
    acc.sum_sq_diff += diff_bb * diff_bb;
}

fn acc_to_result(acc: &ArenaAcc) -> ArenaMatchResult {
    let n = acc.hands.max(1) as f64;
    let mean = acc.sum_diff / n;
    let var = (acc.sum_sq_diff / n - mean * mean).max(0.0);
    let se = (var / n).sqrt();
    let bb_per_100 = mean * 100.0;
    let ci95 = 1.96 * se * 100.0;

    let trash_shove_pct = if acc.challenger_trash > 0 {
        (acc.challenger_trash_shoves as f64 * 100.0) / acc.challenger_trash as f64
    } else {
        0.0
    };

    // Điều kiện soán ngôi: Thắng có ý nghĩa thống kê VÀ tỷ lệ Shove bài rác phải dưới 5.0%
    let is_winner = bb_per_100 >= 2.0
        && bb_per_100 > (se * 100.0 * 1.25)
        && trash_shove_pct <= 5.0;

    ArenaMatchResult {
        hands: acc.hands,
        challenger_bb_per_100: bb_per_100,
        ci95,
        challenger_trash_shove_pct: trash_shove_pct,
        is_winner,
    }
}

/// Đấu trường mặc định 8-max (tương thích ngược với các caller cũ)
pub fn evaluate_against_champion(
    challenger: &dyn Policy,
    champion: &dyn Policy,
    total_hands: usize,
    stack_bb: u32,
) -> ArenaMatchResult {
    evaluate_against_champion_table(challenger, champion, total_hands, stack_bb, ARENA_PLAYERS)
}

/// Đấu trường đối đầu Champion tùy biến số người chơi trên bàn (6-max hoặc 8-max)
pub fn evaluate_against_champion_table(
    challenger: &dyn Policy,
    champion: &dyn Policy,
    total_hands: usize,
    stack_bb: u32,
    num_players: usize,
) -> ArenaMatchResult {
    let opps = [champion];
    let acc = (0..total_hands)
        .into_par_iter()
        .fold(ArenaAcc::default, |mut acc, _| {
            let mut rng = thread_rng();
            // Đánh xen kẽ cả mức stack_bb (100BB) và Push/Fold (12BB) trong Arena để kiểm duyệt toàn diện
            let actual_depth = if rng.gen_bool(0.20) { 12 } else { stack_bb };
            play_arena_team_hand(challenger, &opps, actual_depth, &mut acc, &mut rng, num_players);
            acc
        })
        .reduce(ArenaAcc::default, |a, b| a.merge(&b));

    acc_to_result(&acc)
}

/// Đấu trường Cash Game thực chiến (mô phỏng bàn 6-max PokerStars có tích lũy chip và Deep Stack)
pub fn evaluate_against_champion_cash(
    challenger: &dyn Policy,
    champion: &dyn Policy,
    total_hands: usize,
    num_players: usize,
) -> ArenaMatchResult {
    let opps = [champion];
    let acc = (0..total_hands)
        .into_par_iter()
        .fold(ArenaAcc::default, |mut acc, _| {
            let mut rng = thread_rng();
            play_arena_cash_hand(challenger, &opps, &mut acc, &mut rng, num_players);
            acc
        })
        .reduce(ArenaAcc::default, |a, b| a.merge(&b));

    acc_to_result(&acc)
}

fn play_arena_cash_hand<R: Rng>(
    challenger: &dyn Policy,
    opponents: &[&dyn Policy],
    acc: &mut ArenaAcc,
    rng: &mut R,
    num_players: usize,
) {
    let bb = 100u32;
    let sb = 50u32;
    let n = num_players.clamp(2, 8);
    let stacks = crate::sampler::TournamentStackSampler::sample_cash_game_stacks(rng, n, bb);
    let button = rng.gen_range(0..n);

    let mut env = Environment::new(n, &stacks, bb, sb, button);
    env.state.is_cash_game = true;
    env.reset_hand(rng, button);

    let challenger_seats: Vec<usize> = (0..n).filter(|&seat| seat % 2 == 0).collect();
    let initial_team_chips: f64 = challenger_seats.iter().map(|&seat| stacks[seat] as f64).sum();

    let mut guard = 0;
    while env.state.street != Street::Showdown && guard < 400 {
        guard += 1;
        let cur = env.state.current_player_idx;
        let mask = env.get_action_mask();
        if mask.0 == 0 {
            break;
        }

        let feats = env.state.encode_features(cur);
        let is_challenger = cur.is_multiple_of(2);

        let action = if is_challenger {
            let a = challenger.select_action(&feats, &mask);
            acc.total_actions += 1;
            if a == Action::AllIn {
                acc.challenger_shoves += 1;
            }
            if feats[F_STREET_PREFLOP] > 0.5 && feats[F_HAND_STRENGTH] < TRASH_CHEN {
                acc.challenger_trash += 1;
                if a == Action::AllIn {
                    acc.challenger_trash_shoves += 1;
                }
            }
            a
        } else {
            let opp_idx = (cur / 2) % opponents.len();
            opponents[opp_idx].select_action(&feats, &mask)
        };

        match env.step(action) {
            Ok(true) | Err(_) => break,
            Ok(false) => {}
        }
    }

    let final_team_chips: f64 = challenger_seats
        .iter()
        .map(|&seat| env.state.players[seat].stack as f64)
        .sum();

    let num_challengers = challenger_seats.len().max(1) as f64;
    let diff_bb = (final_team_chips - initial_team_chips) / (bb as f64) / num_challengers;

    acc.hands += 1;
    acc.sum_diff += diff_bb;
    acc.sum_sq_diff += diff_bb * diff_bb;
}

/// Đấu trường 8-max đối đầu Biệt đội Heuristic All-Star:
/// Ghế 0, 2, 4, 6: Challenger
/// Ghế 1: TightAggressive | Ghế 3: LagShark | Ghế 5: NashPushBot | Ghế 7: IcmSniper
pub fn evaluate_against_heuristic_team(
    challenger: &dyn Policy,
    total_hands: usize,
    stack_bb: u32,
) -> ArenaMatchResult {
    let tag = TightAggressive;
    let lag = LagShark;
    let nash = NashPushBot;
    let icm = IcmSniper;
    let opps: [&dyn Policy; 4] = [&tag, &lag, &nash, &icm];

    let acc = (0..total_hands)
        .into_par_iter()
        .fold(ArenaAcc::default, |mut acc, _| {
            let mut rng = thread_rng();
            let actual_depth = if rng.gen_bool(0.20) { 12 } else { stack_bb };
            play_arena_team_hand(challenger, &opps, actual_depth, &mut acc, &mut rng, ARENA_PLAYERS);
            acc
        })
        .reduce(ArenaAcc::default, |a, b| a.merge(&b));

    acc_to_result(&acc)
}

/// Đánh giá toàn diện trên Đấu trường Arena 8-max với từng bot Heuristic và Biệt đội hỗn hợp
pub fn evaluate_arena_suite(
    challenger: &dyn Policy,
    hands_per_match: usize,
    stack_bb: u32,
) -> Vec<(&'static str, ArenaMatchResult)> {
    vec![
        ("Arena_TAG", evaluate_against_champion(challenger, &TightAggressive, hands_per_match, stack_bb)),
        ("Arena_LagShark", evaluate_against_champion(challenger, &LagShark, hands_per_match, stack_bb)),
        ("Arena_NashPush", evaluate_against_champion(challenger, &NashPushBot, hands_per_match, stack_bb)),
        ("Arena_IcmSniper", evaluate_against_champion(challenger, &IcmSniper, hands_per_match, stack_bb)),
        ("Arena_HeuristicTeam", evaluate_against_heuristic_team(challenger, hands_per_match, stack_bb)),
    ]
}

pub fn print_arena_results(label: &str, results: &[(&str, ArenaMatchResult)]) {
    println!("   ┌─ Arena Benchmark: {}", label);
    println!("   │ {:<20} {:>8} {:>10} {:>8} {:>12} {:>8}",
        "Opponent", "Hands", "bb/100", "±95%", "TrashShove%", "Status");
    for (name, r) in results {
        let status = if r.is_winner { "WIN 👑" } else { "DEF 🛡️" };
        println!("   │ {:<20} {:>8} {:>+10.1} {:>8.1} {:>11.1}% {:>8}",
            name, r.hands, r.challenger_bb_per_100, r.ci95, r.challenger_trash_shove_pct, status);
    }
    println!("   └─");
}

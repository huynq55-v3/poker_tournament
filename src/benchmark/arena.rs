use crate::benchmark::{
    BountyHunter, IcmSniper, LagShark, NashPushBot, NitTrap, StickyFish, ThreeBetAggro,
    TightAggressive,
};
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

/// Đấu trường Final Table Tournament (bàn 4-max, Chip-EV thuần, random stack Dirichlet bất đối xứng)
pub fn evaluate_against_champion_final_table(
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
            play_arena_final_table_hand(challenger, &opps, &mut acc, &mut rng, num_players);
            acc
        })
        .reduce(ArenaAcc::default, |a, b| a.merge(&b));

    acc_to_result(&acc)
}

fn play_arena_final_table_hand<R: Rng>(
    challenger: &dyn Policy,
    opponents: &[&dyn Policy],
    acc: &mut ArenaAcc,
    rng: &mut R,
    num_players: usize,
) {
    let bb = 100u32;
    let sb = 50u32;
    let n = num_players.clamp(2, 8);
    let stacks = if rng.gen_bool(0.75) {
        crate::sampler::TournamentStackSampler::sample_dirichlet_stacks(rng, n, bb)
    } else {
        crate::sampler::TournamentStackSampler::sample_equal_stacks(rng, n, bb)
    };
    let button = rng.gen_range(0..n);

    let mut env = Environment::new(n, &stacks, bb, sb, button);
    env.state.is_cash_game = false; // Thuần Chip-EV Tournament, không trừ phế sàn (No Rake)
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

/// Đấu trường 8-max đối đầu Biệt đội Heuristic All-Star (luân phiên cả 8 trường phái quân xanh):
/// Ghế 0, 2, 4, 6: Challenger
/// Ghế 1, 3, 5, 7: Luân phiên [&tag, &lag, &nash, &icm, &nit, &fish, &aggro, &bounty]
pub fn evaluate_against_heuristic_team(
    challenger: &dyn Policy,
    total_hands: usize,
    stack_bb: u32,
) -> ArenaMatchResult {
    let tag = TightAggressive;
    let lag = LagShark;
    let nash = NashPushBot;
    let icm = IcmSniper;
    let nit = NitTrap;
    let fish = StickyFish;
    let aggro = ThreeBetAggro;
    let bounty = BountyHunter;
    let opps: [&dyn Policy; 8] = [&tag, &lag, &nash, &icm, &nit, &fish, &aggro, &bounty];

    let acc = (0..total_hands)
        .into_par_iter()
        .fold(ArenaAcc::default, |mut acc, _| {
            let mut rng = thread_rng();
            let actual_depth = if rng.gen_bool(0.20) { 12 } else { stack_bb };
            let shift = rng.gen_range(0..8);
            let rotated_opps: [&dyn Policy; 8] = [
                opps[shift % 8],
                opps[(shift + 1) % 8],
                opps[(shift + 2) % 8],
                opps[(shift + 3) % 8],
                opps[(shift + 4) % 8],
                opps[(shift + 5) % 8],
                opps[(shift + 6) % 8],
                opps[(shift + 7) % 8],
            ];
            play_arena_team_hand(challenger, &rotated_opps, actual_depth, &mut acc, &mut rng, ARENA_PLAYERS);
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
        ("Arena_NitTrap", evaluate_against_champion(challenger, &NitTrap, hands_per_match, stack_bb)),
        ("Arena_StickyFish", evaluate_against_champion(challenger, &StickyFish, hands_per_match, stack_bb)),
        ("Arena_ThreeBetAggro", evaluate_against_champion(challenger, &ThreeBetAggro, hands_per_match, stack_bb)),
        ("Arena_BountyHunter", evaluate_against_champion(challenger, &BountyHunter, hands_per_match, stack_bb)),
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

// =========================================================================
// TOURNAMENT SIT & GO FULL ARENA SIMULATION (ĐÁNH ĐẾN KHI HẾT TOUR)
// =========================================================================

pub fn get_tournament_blinds(level: usize) -> (u32, u32) {
    const BLINDS: [(u32, u32); 12] = [
        (50, 100),
        (75, 150),
        (100, 200),
        (150, 300),
        (200, 400),
        (300, 600),
        (500, 1000),
        (800, 1600),
        (1200, 2400),
        (2000, 4000),
        (3000, 6000),
        (5000, 10000),
    ];
    let idx = level.min(BLINDS.len() - 1);
    BLINDS[idx]
}

#[derive(Default, Clone)]
pub struct TournamentAcc {
    pub tournaments: u64,
    pub challenger_first_places: u64,
    pub champion_first_places: u64,
    pub challenger_itm_count: u64,
    pub champion_itm_count: u64,
    pub challenger_rank_sum: f64,
    pub champion_rank_sum: f64,
    pub challenger_payout_sum: f64,
    pub champion_payout_sum: f64,
    pub challenger_shoves: u64,
    pub challenger_trash: u64,
    pub challenger_trash_shoves: u64,
    pub total_actions: u64,
    pub total_hands: u64,
}

impl TournamentAcc {
    pub fn merge(mut self, o: &TournamentAcc) -> TournamentAcc {
        self.tournaments += o.tournaments;
        self.challenger_first_places += o.challenger_first_places;
        self.champion_first_places += o.champion_first_places;
        self.challenger_itm_count += o.challenger_itm_count;
        self.champion_itm_count += o.champion_itm_count;
        self.challenger_rank_sum += o.challenger_rank_sum;
        self.champion_rank_sum += o.champion_rank_sum;
        self.challenger_payout_sum += o.challenger_payout_sum;
        self.champion_payout_sum += o.champion_payout_sum;
        self.challenger_shoves += o.challenger_shoves;
        self.challenger_trash += o.challenger_trash;
        self.challenger_trash_shoves += o.challenger_trash_shoves;
        self.total_actions += o.total_actions;
        self.total_hands += o.total_hands;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct TournamentArenaResult {
    pub tournaments: usize,
    pub challenger_first_places: usize,
    pub champion_first_places: usize,
    pub challenger_itm_count: usize,
    pub champion_itm_count: usize,
    pub challenger_avg_rank: f64,
    pub champion_avg_rank: f64,
    pub challenger_avg_icm_payout: f64,
    pub challenger_trash_shove_pct: f64,
    pub total_hands: usize,
    pub avg_hands_per_tour: f64,
    pub is_winner: bool,
}

fn play_single_full_tournament<R: Rng>(
    challenger: &dyn Policy,
    champion: &dyn Policy,
    num_players: usize,
    initial_bb: u32,
    rng: &mut R,
    acc: &mut TournamentAcc,
) {
    let n = num_players.clamp(2, 8);
    let base_bb = 100u32;
    let base_sb = 50u32;
    let stacks = vec![initial_bb * base_bb; n];
    let mut button = rng.gen_range(0..n);

    let mut env = Environment::new(n, &stacks, base_bb, base_sb, button);
    env.state.is_cash_game = false; // Sit & Go Tournament: No rake, pure ICM

    let mut finish_ranks = vec![0usize; n];
    let mut eliminated_count = 0;
    let mut hand_idx = 0;

    while eliminated_count < n - 1 && hand_idx < 300 {
        let alive_before: Vec<usize> = (0..n).filter(|&i| env.state.players[i].stack > 0).collect();
        if alive_before.len() <= 1 {
            break;
        }

        let mut stacks_before = [0u32; 8];
        for &i in &alive_before {
            stacks_before[i] = env.state.players[i].stack;
        }

        // Tăng Blind Level định kỳ sau mỗi n ván
        let blind_level = hand_idx / n;
        let (sb, bb) = get_tournament_blinds(blind_level);
        env.state.sb_size = sb;
        env.state.bb_size = bb;

        // Chuyển button tới người chơi còn sống sót tiếp theo
        let mut next_btn = (button + 1) % n;
        while env.state.players[next_btn].stack == 0 {
            next_btn = (next_btn + 1) % n;
        }
        button = next_btn;
        env.reset_hand(rng, button);

        // Chơi ván bài
        let mut guard = 0;
        while env.state.street != Street::Showdown && guard < 400 {
            guard += 1;
            let cur = env.state.current_player_idx;
            let mask = env.get_action_mask();
            if mask.0 == 0 {
                break;
            }

            let feats = env.state.encode_features(cur);
            let is_challenger = cur % 2 == 0;

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
                champion.select_action(&feats, &mask)
            };

            match env.step(action) {
                Ok(true) | Err(_) => break,
                Ok(false) => {}
            }
        }

        hand_idx += 1;

        // Ghi nhận những người vừa bị loại ở ván này
        let mut newly_busted: Vec<usize> = alive_before
            .into_iter()
            .filter(|&i| env.state.players[i].stack == 0)
            .collect();

        if !newly_busted.is_empty() {
            newly_busted.sort_by_key(|&i| stacks_before[i]);
            for busted_player in newly_busted {
                let rank = n - eliminated_count;
                finish_ranks[busted_player] = rank;
                eliminated_count += 1;
            }
        }
    }

    // Người còn chip duy nhất là Nhà Vô Địch (Hạng 1)!
    if let Some(winner) = (0..n).find(|&i| env.state.players[i].stack > 0 && finish_ranks[i] == 0) {
        finish_ranks[winner] = 1;
    } else {
        let mut remaining: Vec<usize> = (0..n).filter(|&i| finish_ranks[i] == 0).collect();
        remaining.sort_by_key(|&i| std::cmp::Reverse(env.state.players[i].stack));
        for (idx, &player) in remaining.iter().enumerate() {
            finish_ranks[player] = idx + 1;
        }
    }

    // Thống kê kết quả giải đấu
    let payouts = crate::equity::icm::get_sng_payouts(n);
    let winner = (0..n).find(|&i| finish_ranks[i] == 1).unwrap_or(0);
    if winner % 2 == 0 {
        acc.challenger_first_places += 1;
    } else {
        acc.champion_first_places += 1;
    }

    for i in 0..n {
        let rank = finish_ranks[i];
        let payout = if rank >= 1 && rank <= payouts.len() {
            payouts[rank - 1]
        } else {
            0.0
        };

        if i % 2 == 0 {
            acc.challenger_rank_sum += rank as f64;
            acc.challenger_payout_sum += payout as f64;
            if rank <= 3 {
                acc.challenger_itm_count += 1;
            }
        } else {
            acc.champion_rank_sum += rank as f64;
            acc.champion_payout_sum += payout as f64;
            if rank <= 3 {
                acc.champion_itm_count += 1;
            }
        }
    }

    acc.total_hands += hand_idx as u64;
    acc.tournaments += 1;
}

/// Đấu trường Tournament đối đầu Champion: Chạy nhiều giải đấu Sit & Go hoàn chỉnh (8 người, 50 BB ban đầu, đánh đến khi hết tour)
pub fn evaluate_against_champion_tournament(
    challenger: &dyn Policy,
    champion: &dyn Policy,
    total_tournaments: usize,
    initial_bb: u32,
    num_players: usize,
) -> TournamentArenaResult {
    let acc = (0..total_tournaments)
        .into_par_iter()
        .fold(TournamentAcc::default, |mut acc, _| {
            let mut rng = thread_rng();
            play_single_full_tournament(challenger, champion, num_players, initial_bb, &mut rng, &mut acc);
            acc
        })
        .reduce(TournamentAcc::default, |a, b| a.merge(&b));

    let t = acc.tournaments.max(1) as f64;
    let num_challenger_seats = ((num_players + 1) / 2) as f64;
    let num_champion_seats = (num_players / 2) as f64;

    let challenger_avg_rank = acc.challenger_rank_sum / (t * num_challenger_seats);
    let champion_avg_rank = acc.champion_rank_sum / (t * num_champion_seats);
    let challenger_avg_icm = acc.challenger_payout_sum / (t * num_challenger_seats);

    let trash_shove_pct = if acc.challenger_trash > 0 {
        (acc.challenger_trash_shoves as f64 * 100.0) / acc.challenger_trash as f64
    } else {
        0.0
    };

    // Điều kiện soán ngôi: Thứ hạng trung bình tốt hơn (nhỏ hơn), Payout ICM dương, và TrashShove <= 5%
    let is_winner = challenger_avg_rank < (champion_avg_rank - 0.05)
        && challenger_avg_icm > 0.01
        && acc.challenger_first_places >= acc.champion_first_places
        && trash_shove_pct <= 5.0;

    TournamentArenaResult {
        tournaments: acc.tournaments as usize,
        challenger_first_places: acc.challenger_first_places as usize,
        champion_first_places: acc.champion_first_places as usize,
        challenger_itm_count: acc.challenger_itm_count as usize,
        champion_itm_count: acc.champion_itm_count as usize,
        challenger_avg_rank,
        champion_avg_rank,
        challenger_avg_icm_payout: challenger_avg_icm,
        challenger_trash_shove_pct: trash_shove_pct,
        total_hands: acc.total_hands as usize,
        avg_hands_per_tour: acc.total_hands as f64 / t,
        is_winner,
    }
}

/// Giả lập 1 giải đấu Tournament chi tiết từ đầu đến khi hết tour và in log tường thuật
pub fn simulate_demo_tournament<R: Rng>(
    policy: &dyn Policy,
    num_players: usize,
    initial_bb: u32,
    rng: &mut R,
) -> (usize, usize) {
    let n = num_players.clamp(2, 8);
    let base_bb = 100u32;
    let base_sb = 50u32;
    let stacks = vec![initial_bb * base_bb; n];
    let mut button = rng.gen_range(0..n);

    let mut env = Environment::new(n, &stacks, base_bb, base_sb, button);
    env.state.is_cash_game = false;

    println!("🏆 [KHỞI TRANH TOURNAMENT {}-MAX] Mỗi người cầm {} chip ({} BB). Tổng chip bàn: {} chip.",
        n, initial_bb * base_bb, initial_bb, n as u32 * initial_bb * base_bb);

    let mut finish_ranks = vec![0usize; n];
    let mut eliminated_count = 0;
    let mut hand_idx = 0;

    while eliminated_count < n - 1 && hand_idx < 300 {
        let alive_before: Vec<usize> = (0..n).filter(|&i| env.state.players[i].stack > 0).collect();
        if alive_before.len() <= 1 {
            break;
        }

        let mut stacks_before = [0u32; 8];
        for &i in &alive_before {
            stacks_before[i] = env.state.players[i].stack;
        }

        let blind_level = hand_idx / n;
        let (sb, bb) = get_tournament_blinds(blind_level);
        env.state.sb_size = sb;
        env.state.bb_size = bb;

        let mut next_btn = (button + 1) % n;
        while env.state.players[next_btn].stack == 0 {
            next_btn = (next_btn + 1) % n;
        }
        button = next_btn;
        env.reset_hand(rng, button);

        // Tường thuật ngắn gọn các hand quan trọng
        if hand_idx == 0 || alive_before.len() == 5 || alive_before.len() == 3 || alive_before.len() == 2 {
            let stage = match alive_before.len() {
                2 => "⚔️  [HEADS-UP FINAL]",
                3 => "🎉 [IN THE MONEY - TOP 3]",
                5 => "⚡ [BUBBLE STAGE]",
                _ => "🟢 [FULL TABLE]",
            };
            println!("   --- Hand #{:02} {} ({} người còn lại | Blinds: {}/{}) ---",
                hand_idx + 1, stage, alive_before.len(), sb, bb);
        }

        let mut guard = 0;
        while env.state.street != Street::Showdown && guard < 400 {
            guard += 1;
            let cur = env.state.current_player_idx;
            let mask = env.get_action_mask();
            if mask.0 == 0 {
                break;
            }
            let feats = env.state.encode_features(cur);
            let action = policy.select_action(&feats, &mask);
            match env.step(action) {
                Ok(true) | Err(_) => break,
                Ok(false) => {}
            }
        }

        hand_idx += 1;

        let mut newly_busted: Vec<usize> = alive_before
            .into_iter()
            .filter(|&i| env.state.players[i].stack == 0)
            .collect();

        if !newly_busted.is_empty() {
            newly_busted.sort_by_key(|&i| stacks_before[i]);
            for busted_player in newly_busted {
                let rank = n - eliminated_count;
                finish_ranks[busted_player] = rank;
                eliminated_count += 1;
                println!("   💀 [LOẠI KHỎI GIẢI] Ghế #{} hết chip và bị loại ở HẠNG {}! (Bàn còn {} người)",
                    busted_player, rank, n - eliminated_count);
            }
        }
    }

    let winner = (0..n).find(|&i| env.state.players[i].stack > 0 && finish_ranks[i] == 0)
        .or_else(|| (0..n).max_by_key(|&i| env.state.players[i].stack))
        .unwrap_or(0);
    finish_ranks[winner] = 1;

    println!("   🏆 TOURNAMENT KẾT THÚC sau {} ván! Ghế #{} ôm trọn toàn bộ {} chip và trở thành NHÀ VÔ ĐỊCH! 👑\n",
        hand_idx, winner, n as u32 * initial_bb * base_bb);

    (winner, hand_idx)
}

/// Giả lập 1 giải đấu Sit & Go với Hero (Ghế #0) đối đầu với các bot Baseline (TAG, LAG, Push/Fold, IcmSniper)
/// Đánh liên tục cho đến khi chỉ còn 1 người vô địch duy nhất ("hết tour")
pub fn simulate_hero_tournament<R: Rng>(
    hero: &dyn Policy,
    num_players: usize,
    initial_bb: u32,
    rng: &mut R,
) -> (usize, usize) {
    let n = num_players.clamp(2, 8);
    let base_bb = 100u32;
    let base_sb = 50u32;
    let stacks = vec![initial_bb * base_bb; n];
    let mut button = rng.gen_range(0..n);

    let mut env = Environment::new(n, &stacks, base_bb, base_sb, button);
    env.state.is_cash_game = false;

    let tag = TightAggressive;
    let lag = LagShark;
    let push = NashPushBot;
    let sniper = IcmSniper;
    let nit = NitTrap;
    let fish = StickyFish;
    let aggro = ThreeBetAggro;
    let bounty = BountyHunter;
    let opp_policies: [&dyn Policy; 8] = [&tag, &lag, &push, &sniper, &nit, &fish, &aggro, &bounty];

    let mut finish_ranks = vec![0usize; n];
    let mut eliminated_count = 0;
    let mut hand_idx = 0;

    while eliminated_count < n - 1 && hand_idx < 300 {
        let alive_before: Vec<usize> = (0..n).filter(|&i| env.state.players[i].stack > 0).collect();
        if alive_before.len() <= 1 {
            break;
        }

        let mut stacks_before = [0u32; 8];
        for &i in &alive_before {
            stacks_before[i] = env.state.players[i].stack;
        }

        let blind_level = hand_idx / n;
        let (sb, bb) = get_tournament_blinds(blind_level);
        env.state.sb_size = sb;
        env.state.bb_size = bb;

        let mut next_btn = (button + 1) % n;
        while env.state.players[next_btn].stack == 0 {
            next_btn = (next_btn + 1) % n;
        }
        button = next_btn;
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
            let action = if cur == 0 {
                hero.select_action(&feats, &mask)
            } else {
                opp_policies[(cur - 1) % opp_policies.len()].select_action(&feats, &mask)
            };
            match env.step(action) {
                Ok(true) | Err(_) => break,
                Ok(false) => {}
            }
        }

        hand_idx += 1;

        let mut newly_busted: Vec<usize> = alive_before
            .into_iter()
            .filter(|&i| env.state.players[i].stack == 0)
            .collect();

        if !newly_busted.is_empty() {
            newly_busted.sort_by_key(|&i| stacks_before[i]);
            for busted_player in newly_busted {
                let rank = n - eliminated_count;
                finish_ranks[busted_player] = rank;
                eliminated_count += 1;
            }
        }
    }

    if let Some(winner) = (0..n).find(|&i| env.state.players[i].stack > 0 && finish_ranks[i] == 0) {
        finish_ranks[winner] = 1;
    } else {
        let mut remaining: Vec<usize> = (0..n).filter(|&i| finish_ranks[i] == 0).collect();
        remaining.sort_by_key(|&i| std::cmp::Reverse(env.state.players[i].stack));
        for (idx, &player) in remaining.iter().enumerate() {
            finish_ranks[player] = idx + 1;
        }
    }

    (finish_ranks[0], hand_idx)
}


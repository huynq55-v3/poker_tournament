use crate::environment::action::Action;
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::self_play::Policy;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

const F_STREET_PREFLOP: usize = 119;
const F_HAND_STRENGTH: usize = 163;
const TRASH_CHEN: f32 = 0.30;
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

fn play_arena_hand<R: Rng>(
    challenger: &dyn Policy,
    champion: &dyn Policy,
    stack_bb: u32,
    acc: &mut ArenaAcc,
    rng: &mut R,
) {
    let bb = 100u32;
    let sb = 50u32;
    let n = ARENA_PLAYERS;
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

    let initial_team_chips = (4 * stack_bb * bb) as f64;
    let final_team_chips: f64 = [0, 2, 4, 6]
        .iter()
        .map(|&seat| env.state.players[seat].stack as f64)
        .sum();

    let diff_bb = (final_team_chips - initial_team_chips) / (bb as f64) / 4.0;

    acc.hands += 1;
    acc.sum_diff += diff_bb;
    acc.sum_sq_diff += diff_bb * diff_bb;
}

pub fn evaluate_against_champion(
    challenger: &dyn Policy,
    champion: &dyn Policy,
    total_hands: usize,
    stack_bb: u32,
) -> ArenaMatchResult {
    let acc = (0..total_hands)
        .into_par_iter()
        .fold(ArenaAcc::default, |mut acc, _| {
            let mut rng = thread_rng();
            play_arena_hand(challenger, champion, stack_bb, &mut acc, &mut rng);
            acc
        })
        .reduce(ArenaAcc::default, |a, b| a.merge(&b));

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

    let is_winner = bb_per_100 >= 2.0 && bb_per_100 > (se * 100.0 * 1.25);

    ArenaMatchResult {
        hands: acc.hands,
        challenger_bb_per_100: bb_per_100,
        ci95,
        challenger_trash_shove_pct: trash_shove_pct,
        is_winner,
    }
}

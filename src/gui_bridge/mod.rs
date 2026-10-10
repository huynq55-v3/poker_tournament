use crate::environment::action::{ActionMask, NUM_ACTIONS};
use crate::environment::state::{GameState, Position, Street};
use crate::self_play::Policy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerViewModel {
    pub seat_idx: usize,
    pub name: String,
    pub position: String,
    pub stack: u32,
    pub stack_bb: f32,
    pub current_bet: u32,
    pub hole_cards: Vec<String>,
    pub is_current_actor: bool,
    pub is_active: bool,
    pub is_all_in: bool,
    pub is_folded: bool,
    pub is_bot: bool,
    pub gto_probabilities: Option<[f32; NUM_ACTIONS]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableViewModel {
    pub street: String,
    pub pot: u32,
    pub pot_bb: f32,
    pub min_raise: u32,
    pub community_cards: Vec<String>,
    pub players: Vec<PlayerViewModel>,
    pub current_actor_idx: usize,
    pub valid_actions: Vec<String>,
    pub action_logs: Vec<String>,
}

impl TableViewModel {
    pub fn from_game_state<P: Policy>(
        state: &GameState,
        mask: &ActionMask,
        policy: Option<&P>,
        reveal_all_cards: bool,
    ) -> Self {
        let street_str = match state.street {
            Street::Preflop => "Preflop",
            Street::Flop => "Flop",
            Street::Turn => "Turn",
            Street::River => "River",
            Street::Showdown => "Showdown",
        }
        .to_string();

        let board_strs: Vec<String> = state
            .community_cards
            .iter()
            .map(|c| format!("{}", c))
            .collect();

        let mut player_vms = Vec::with_capacity(state.num_players);

        // Lấy danh sách những người thực sự còn sống để gán vị trí chuẩn
        let alive_seats: Vec<usize> = state
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.stack > 0 || p.is_all_in)
            .map(|(idx, _)| idx)
            .collect();

        for (i, p) in state.players.iter().enumerate() {
            let pos = Position::for_seat_alive(i, state.button_idx, &alive_seats);
            let pos_str = format!("{:?}", pos);

            let cards = if reveal_all_cards || !p.is_bot || state.street == Street::Showdown {
                p.hole_cards
                    .iter()
                    .filter_map(|c| c.map(|card| format!("{}", card)))
                    .collect()
            } else if p.is_in_hand() {
                vec!["🂠".to_string(), "🂠".to_string()]
            } else {
                Vec::new()
            };

            let is_actor = i == state.current_player_idx && state.street != Street::Showdown;
            let gto_probs = if is_actor {
                policy.map(|pol| {
                    let feat = state.encode_features(i);
                    pol.get_action_probs(&feat, mask)
                })
            } else {
                None
            };

            player_vms.push(PlayerViewModel {
                seat_idx: i,
                name: if p.is_bot {
                    format!("Bot_{}", i)
                } else {
                    format!("Hero_{}", i)
                },
                position: pos_str,
                stack: p.stack,
                stack_bb: p.stack as f32 / state.bb_size as f32,
                current_bet: p.current_bet,
                hole_cards: cards,
                is_current_actor: is_actor,
                is_active: p.is_active(),
                is_all_in: p.is_all_in,
                is_folded: p.is_folded,
                is_bot: p.is_bot,
                gto_probabilities: gto_probs,
            });
        }

        let valid_acts: Vec<String> = mask
            .valid_actions()
            .into_iter()
            .map(|a| a.name().to_string())
            .collect();

        let action_logs: Vec<String> = state
            .action_history
            .iter()
            .map(|rec| {
                format!(
                    "[{:?}] Player {} -> {} (chips: {})",
                    rec.street,
                    rec.player_idx,
                    rec.action.name(),
                    rec.amount
                )
            })
            .collect();

        TableViewModel {
            street: street_str,
            pot: state.pot,
            pot_bb: state.pot as f32 / state.bb_size as f32,
            min_raise: state.min_raise,
            community_cards: board_strs,
            players: player_vms,
            current_actor_idx: state.current_player_idx,
            valid_actions: valid_acts,
            action_logs,
        }
    }
}

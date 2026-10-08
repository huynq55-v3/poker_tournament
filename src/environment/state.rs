use crate::poker_core::card::Card;
use crate::environment::action::Action;
use serde::{Deserialize, Serialize};

pub const MAX_PLAYERS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
    Showdown = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Position {
    BTN,
    SB,
    BB,
    UTG,
    UTG1,
    MP,
    HJ,
    CO,
}

impl Position {
    pub fn for_seat(seat: usize, button: usize, num_players: usize) -> Self {
        let rel_pos = (seat + num_players - button) % num_players;
        if num_players == 2 {
            // Heads-Up: Button is SB, Other is BB
            if rel_pos == 0 {
                Position::SB // BTN/SB
            } else {
                Position::BB
            }
        } else {
            match rel_pos {
                0 => Position::BTN,
                1 => Position::SB,
                2 => Position::BB,
                3 => Position::UTG,
                4 => if num_players > 5 { Position::UTG1 } else { Position::CO },
                5 => if num_players > 6 { Position::MP } else { Position::CO },
                6 => Position::HJ,
                _ => Position::CO,
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerState {
    pub seat_idx: usize,
    pub stack: u32,
    pub current_bet: u32,
    pub total_contributed: u32,
    pub hole_cards: [Option<Card>; 2],
    pub is_folded: bool,
    pub is_all_in: bool,
    pub is_bot: bool,
}

impl PlayerState {
    pub fn new(seat_idx: usize, stack: u32, is_bot: bool) -> Self {
        Self {
            seat_idx,
            stack,
            current_bet: 0,
            total_contributed: 0,
            hole_cards: [None, None],
            is_folded: false,
            is_all_in: false,
            is_bot,
        }
    }

    #[inline(always)]
    pub fn is_active(&self) -> bool {
        !self.is_folded && !self.is_all_in && self.stack > 0
    }

    #[inline(always)]
    pub fn is_in_hand(&self) -> bool {
        !self.is_folded
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub player_idx: usize,
    pub street: Street,
    pub action: Action,
    pub amount: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub num_players: usize,
    pub players: Vec<PlayerState>,
    pub community_cards: Vec<Card>,
    pub pot: u32,
    pub street: Street,
    pub button_idx: usize,
    pub current_player_idx: usize,
    pub min_raise: u32,
    pub current_highest_bet: u32,
    pub bb_size: u32,
    pub sb_size: u32,
    pub action_history: Vec<ActionRecord>,
}

impl GameState {
    /// Compute effective stack of `player_idx` vs all remaining active opponents in BB
    pub fn effective_stack_bb(&self, player_idx: usize) -> f32 {
        let player_stack = self.players[player_idx].stack + self.players[player_idx].current_bet;
        let mut max_opp_stack = 0u32;
        for (i, opp) in self.players.iter().enumerate() {
            if i != player_idx && opp.is_in_hand() {
                let opp_total = opp.stack + opp.current_bet;
                if opp_total > max_opp_stack {
                    max_opp_stack = opp_total;
                }
            }
        }
        let eff_chips = player_stack.min(max_opp_stack);
        eff_chips as f32 / self.bb_size as f32
    }

    /// Extract feature vector for Deep CFR / PPO Reinforcement Learning
    /// Feature size: 52 (hole) + 52 (board) + 8 (position) + 4 (street) + 1 (pot) + 1 (to_call) + 8 (stacks) = 126 dims
    pub fn encode_features(&self, player_idx: usize) -> Vec<f32> {
        let mut features = Vec::with_capacity(126);

        // 1. Hole cards multi-hot (52 dims)
        let mut hole_vec = [0.0f32; 52];
        for card_opt in &self.players[player_idx].hole_cards {
            if let Some(card) = card_opt {
                hole_vec[card.0 as usize] = 1.0;
            }
        }
        features.extend_from_slice(&hole_vec);

        // 2. Community cards multi-hot (52 dims)
        let mut board_vec = [0.0f32; 52];
        for card in &self.community_cards {
            board_vec[card.0 as usize] = 1.0;
        }
        features.extend_from_slice(&board_vec);

        // 3. Position relative to BTN (8 dims one-hot)
        let rel_pos = (player_idx + self.num_players - self.button_idx) % self.num_players;
        let mut pos_vec = [0.0f32; 8];
        pos_vec[rel_pos] = 1.0;
        features.extend_from_slice(&pos_vec);

        // 4. Street one-hot (4 dims: Preflop, Flop, Turn, River)
        let mut street_vec = [0.0f32; 4];
        let street_idx = (self.street as usize).min(3);
        street_vec[street_idx] = 1.0;
        features.extend_from_slice(&street_vec);

        // 5. Pot size normalized by 100 BB (1 dim)
        features.push((self.pot as f32) / (self.bb_size as f32 * 100.0));

        // 6. Current to-call amount normalized by 20 BB (1 dim)
        let to_call = self.current_highest_bet.saturating_sub(self.players[player_idx].current_bet);
        features.push((to_call as f32) / (self.bb_size as f32 * 20.0));

        // 7. Stack sizes normalized by 100 BB for 8 seats (8 dims)
        for i in 0..MAX_PLAYERS {
            if i < self.num_players {
                let stack = self.players[i].stack;
                features.push((stack as f32) / (self.bb_size as f32 * 100.0));
            } else {
                features.push(0.0);
            }
        }

        features
    }
}

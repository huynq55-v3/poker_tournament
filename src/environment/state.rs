use crate::environment::action::Action;
use crate::poker_core::card::Card;
use serde::{Deserialize, Serialize};
use crate::equity;
use crate::poker_core::evaluator::evaluate_7_cards;

pub const FEATURE_DIM: usize = 172; // 164 + 8 đặc trưng equity/draw/board

pub const MAX_PLAYERS: usize = 8;

/// Số chiều vector đặc trưng (xem encode_features):
/// 5 (sức mạnh bài) + 52 (hole) + 52 (board) + 8 (vị trí) + 2 (số người sống, heads-up)
/// + 4 (street) + 4 (pot, to_call, my_stack, pot_odds) + 28 (7 đối thủ x 4) + 2 (raises, người trong ván)
/// pub const FEATURE_DIM: usize = 157;

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
                Position::SB
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

        /// (category/9, điểm mạnh 0..1). Preflop dùng công thức Chen, postflop dùng evaluator.
    fn hand_strength(&self, player_idx: usize) -> (f32, f32) {
        let p = &self.players[player_idx];
        let (Some(c0), Some(c1)) = (p.hole_cards[0], p.hole_cards[1]) else {
            return (0.0, 0.0);
        };
        if self.community_cards.len() >= 3 {
            let mut cards = self.community_cards.clone();
            cards.push(c0);
            cards.push(c1);
            let rank = evaluate_7_cards(&cards);
            let cat = (rank.0 >> 24) as f32 / 9.0;
            let within = ((rank.0 >> 16) as f32 / 2400.0).min(1.0);
            (cat, within)
        } else {
            (0.0, chen_score(c0, c1))
        }
    }
    
        /// [equity, equity - pot_odds, equity x số người (so với phần chia công bằng),
    ///  flush draw, straight draw, board có đôi, max cùng chất board, lá cao board]
    fn equity_features(&self, player_idx: usize, to_call: u32) -> [f32; 8] {
        let me = &self.players[player_idx];
        let (Some(c0), Some(c1)) = (me.hole_cards[0], me.hole_cards[1]) else {
            return [0.0; 8];
        };
        let in_hand = self.players.iter().filter(|p| p.is_in_hand()).count().max(1);
        let eq = equity::equity_vs_random([c0, c1], &self.community_cards, in_hand - 1);

        let pot_odds = if to_call > 0 {
            to_call as f32 / (self.pot + to_call) as f32
        } else {
            0.0
        };
        let d = equity::board_and_draw_features([c0, c1], &self.community_cards);

        [
            eq,
            (eq - pot_odds).clamp(-1.0, 1.0),
            (eq * in_hand as f32 / 4.0).min(1.0),
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
        ]
    }

    /// Compute effective stack of `player_idx` vs all remaining opponents in BB
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

    /// Vector đặc trưng theo góc nhìn của `player_idx` (độ dài = FEATURE_DIM).
    /// Đối thủ được sắp xếp XOAY theo ghế của người chơi (ghế kế tiếp theo chiều kim đồng hồ
    /// là slot 0) nên mạng không phải học lại cùng một tình huống cho từng ghế tuyệt đối.
    pub fn encode_features(&self, player_idx: usize) -> Vec<f32> {
        let mut f = Vec::with_capacity(FEATURE_DIM);
        let bb = self.bb_size as f32;
        let me = &self.players[player_idx];

        // 0. Sức mạnh bài tẩy (5)
        if let (Some(c0), Some(c1)) = (me.hole_cards[0], me.hole_cards[1]) {
            let r0 = c0.rank_val() as f32;
            let r1 = c1.rank_val() as f32;
            f.push(r0.max(r1) / 12.0);
            f.push(r0.min(r1) / 12.0);
            f.push(if c0.rank() == c1.rank() { 1.0 } else { 0.0 });
            f.push(if c0.suit() == c1.suit() { 1.0 } else { 0.0 });
            f.push((r0 - r1).abs() / 12.0);
        } else {
            f.extend_from_slice(&[0.0; 5]);
        }

        // 1. Hole cards multi-hot (52)
        let mut hole_vec = [0.0f32; 52];
        for card in me.hole_cards.iter().flatten() {
            hole_vec[card.0 as usize] = 1.0;
        }
        f.extend_from_slice(&hole_vec);

        // 2. Community cards multi-hot (52)
        let mut board_vec = [0.0f32; 52];
        for card in &self.community_cards {
            board_vec[card.0 as usize] = 1.0;
        }
        f.extend_from_slice(&board_vec);

        // 3. Vị trí tương đối so với button trong số người còn sống (8) + số người sống + cờ heads-up (2)
        let alive: Vec<usize> = self
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.stack > 0 || p.is_all_in)
            .map(|(i, _)| i)
            .collect();
        let num_alive = alive.len().max(2);
        let btn_pos = alive.iter().position(|&s| s == self.button_idx).unwrap_or(0);
        let my_pos = alive.iter().position(|&s| s == player_idx).unwrap_or(0);
        let rel_pos = (my_pos + num_alive - btn_pos) % num_alive;

        let mut pos_vec = [0.0f32; 8];
        pos_vec[rel_pos.min(7)] = 1.0;
        f.extend_from_slice(&pos_vec);
        f.push(num_alive as f32 / MAX_PLAYERS as f32);
        f.push(if num_alive == 2 { 1.0 } else { 0.0 });

        // 4. Street one-hot (4)
        let mut street_vec = [0.0f32; 4];
        street_vec[(self.street as usize).min(3)] = 1.0;
        f.extend_from_slice(&street_vec);

        // 5. Pot, to_call, stack của mình, pot odds (4)
        let to_call = self.current_highest_bet.saturating_sub(me.current_bet);
        f.push(self.pot as f32 / (bb * 100.0));
        f.push(to_call as f32 / (bb * 20.0));
        f.push(me.stack as f32 / (bb * 100.0));
        let denom = (self.pot + to_call) as f32;
        f.push(if denom > 0.0 { to_call as f32 / denom } else { 0.0 });

        // 6. 7 đối thủ theo thứ tự xoay: stack, bet hiện tại, còn trong ván, all-in (28)
        for k in 1..MAX_PLAYERS {
            if k < self.num_players {
                let o = &self.players[(player_idx + k) % self.num_players];
                f.push(o.stack as f32 / (bb * 100.0));
                f.push(o.current_bet as f32 / (bb * 20.0));
                f.push(if o.is_in_hand() { 1.0 } else { 0.0 });
                f.push(if o.is_all_in { 1.0 } else { 0.0 });
            } else {
                f.extend_from_slice(&[0.0; 4]);
            }
        }

        // 7. Tóm tắt lịch sử: số lần raise trong street hiện tại, số người còn trong ván (2)
        let raises = self
            .action_history
            .iter()
            .filter(|r| {
                r.street == self.street
                    && matches!(
                        r.action,
                        Action::BetPot33 | Action::BetPot66 | Action::BetPot100 | Action::AllIn
                    )
            })
            .count();
        f.push((raises as f32 / 4.0).min(1.0));
        let in_hand = self.players.iter().filter(|p| p.is_in_hand()).count();
        f.push(in_hand as f32 / MAX_PLAYERS as f32);
        
                // 8. Ngữ cảnh push/fold & sức mạnh bài (7)
        let eff_bb = self.effective_stack_bb(player_idx);
        f.push((eff_bb / 100.0).min(1.0));
        f.push(if eff_bb < 12.0 { 1.0 } else { 0.0 });
        f.push(if to_call > 0 { 1.0 } else { 0.0 });
        let my_stack = me.stack as f32;
        f.push(if my_stack > 0.0 { (to_call as f32 / my_stack).min(1.0) } else { 1.0 });
        let pot_after = (self.pot + to_call) as f32;
        f.push(if pot_after > 0.0 { (my_stack / pot_after / 20.0).min(1.0) } else { 0.0 }); // SPR
        let (cat, strength) = self.hand_strength(player_idx);
        f.push(cat);
        f.push(strength);
        
                // 9. Equity & draw & board texture (8)
        f.extend_from_slice(&self.equity_features(player_idx, to_call));

                assert_eq!(f.len(), FEATURE_DIM, "encode_features length mismatch");
        f
    }
}

/// Công thức Chen chuẩn hóa về 0..1
fn chen_score(c0: Card, c1: Card) -> f32 {
    fn pts(r: u8) -> f32 {
        match r {
            12 => 10.0,
            11 => 8.0,
            10 => 7.0,
            9 => 6.0,
            _ => (r as f32 + 2.0) / 2.0,
        }
    }
    let (hi, lo) = if c0.rank_val() >= c1.rank_val() {
        (c0.rank_val(), c1.rank_val())
    } else {
        (c1.rank_val(), c0.rank_val())
    };
    let mut s = pts(hi);
    if hi == lo {
        s = (s * 2.0).max(5.0);
    } else {
        if c0.suit() == c1.suit() {
            s += 2.0;
        }
        let gap = hi - lo - 1;
        s -= match gap {
            0 => 0.0,
            1 => 1.0,
            2 => 2.0,
            3 => 4.0,
            _ => 5.0,
        };
        if gap <= 1 && hi < 10 {
            s += 1.0;
        }
    }
    ((s + 1.0) / 21.0).clamp(0.0, 1.0)
}

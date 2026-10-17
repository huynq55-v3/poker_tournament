use crate::environment::action::Action;
use crate::poker_core::card::Card;
use serde::{Deserialize, Serialize};
use crate::equity;
use crate::poker_core::evaluator::evaluate_7_cards;
use std::cell::RefCell;

pub const FEATURE_DIM: usize = 176;
pub const MAX_PLAYERS: usize = 8;

// Bộ nhớ đệm Thread-Local siêu tốc cho Equity & Draw Features (tránh tính lại Monte Carlo trên cùng 1 Street)
thread_local! {
    static EQUITY_CACHE: RefCell<(u64, f32, [f32; 5])> = const { RefCell::new((0, 0.0, [0.0; 5])) };
}

#[inline(always)]
fn compute_cache_key(c0: Card, c1: Card, board: &[Card], num_opp: usize) -> u64 {
    let mut k = ((c0.0 as u64) << 8) | (c1.0 as u64) | ((num_opp as u64) << 16) | ((board.len() as u64) << 20);
    for (i, b) in board.iter().enumerate() {
        k ^= (b.0 as u64 + 1) << (24 + i * 7);
    }
    // Đảm bảo key != 0
    k | 0x8000_0000_0000_0000
}

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
    pub fn for_seat_alive(seat: usize, button: usize, alive_seats: &[usize]) -> Self {
        let num_alive = alive_seats.len().max(2);
        let btn_pos = alive_seats.iter().position(|&s| s == button).unwrap_or(0);
        let my_pos = alive_seats.iter().position(|&s| s == seat).unwrap_or(0);
        let rel_pos = (my_pos + num_alive - btn_pos) % num_alive;

        if num_alive == 2 {
            if rel_pos == 0 { Position::SB } else { Position::BB }
        } else {
            match rel_pos {
                0 => Position::BTN,
                1 => Position::SB,
                2 => Position::BB,
                3 => Position::UTG,
                4 => if num_alive > 5 { Position::UTG1 } else { Position::CO },
                5 => if num_alive > 6 { Position::MP } else { Position::CO },
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
    #[serde(default)]
    pub is_cash_game: bool,
}

impl GameState {
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

    fn equity_features(&self, player_idx: usize, to_call: u32) -> [f32; 8] {
        let me = &self.players[player_idx];
        let (Some(c0), Some(c1)) = (me.hole_cards[0], me.hole_cards[1]) else {
            return [0.0; 8];
        };
        let in_hand = self.players.iter().filter(|p| p.is_in_hand()).count().max(1);
        let num_opp = in_hand.saturating_sub(1);

        let key = compute_cache_key(c0, c1, &self.community_cards, num_opp);
        let (eq, d) = EQUITY_CACHE.with(|cache| {
            let mut c = cache.borrow_mut();
            if c.0 == key {
                (c.1, c.2)
            } else {
                let eq_val = equity::equity_vs_random([c0, c1], &self.community_cards, num_opp);
                let draw_val = equity::board_and_draw_features([c0, c1], &self.community_cards);
                *c = (key, eq_val, draw_val);
                (eq_val, draw_val)
            }
        });

        let pot_odds = if to_call > 0 {
            to_call as f32 / (self.pot + to_call) as f32
        } else {
            0.0
        };

        // Tìm người chơi vừa thực hiện hành động Bet/Raise cuối cùng (Last Aggressor)
        // và đếm xem người đó đã chủ động bet/raise bao nhiêu vòng (Streets) trong ván này.
        let last_aggressor = self
            .action_history
            .iter()
            .rev()
            .find(|r| {
                matches!(
                    r.action,
                    Action::BetSmall | Action::BetPot33 | Action::BetPot75 | Action::AllIn
                )
            })
            .map(|r| r.player_idx);

        let aggressor_street_barrels = if let Some(agg_idx) = last_aggressor {
            if agg_idx != player_idx {
                let mut streets_mask = 0u8;
                for r in &self.action_history {
                    if r.player_idx == agg_idx
                        && matches!(
                            r.action,
                            Action::BetSmall | Action::BetPot33 | Action::BetPot75 | Action::AllIn
                        )
                    {
                        streets_mask |= 1 << (r.street as u8);
                    }
                }
                (streets_mask.count_ones() as f32 / 4.0).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        [
            eq,                                 // f[168]: Equity thô khách quan (0..1)
            (eq - pot_odds).clamp(-1.0, 1.0),   // f[169]: Chênh lệch Equity so với Pot Odds
            aggressor_street_barrels,           // f[170]: Số vòng cược mà Last Aggressor đã barrel (0.25 = 1 vòng, 0.5 = 2 vòng, 0.75 = 3 vòng, 1.0 = 4 vòng)
            d[0],                               // f[171]: Flush draw
            d[1],                               // f[172]: Straight draw
            d[2],                               // f[173]: Paired board
            d[3],                               // f[174]: Max suit on board / 5
            d[4],                               // f[175]: Board high card / 12
        ]
    }

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

    /// Mã hóa vector đặc trưng chuẩn xác:
    /// Nén và đồng bộ tuyệt đối giữa số người còn sống thực tế và mạng lúc train.
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

        // 3. Vị trí và trạng thái sống còn thực tế (8 + 2 + 4 Tournament Features = 14)
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
        if num_alive == 2 {
            // Heads-Up: BTN (rel_pos == 0) vừa là BTN vừa là SB -> pos_vec[0] = 1.0 (BTN).
            // Người còn lại (rel_pos == 1) là Big Blind -> pos_vec[2] = 1.0 (BB), tránh nhầm với SB của 8-max.
            if rel_pos == 0 {
                pos_vec[0] = 1.0;
            } else {
                pos_vec[2] = 1.0;
            }
        } else {
            pos_vec[rel_pos.min(7)] = 1.0;
        }
        f.extend_from_slice(&pos_vec);
        f.push(num_alive as f32 / MAX_PLAYERS as f32);
        f.push(if num_alive == 2 { 1.0 } else { 0.0 });

        // 🟢 CÁC ĐẶC TRƯNG TOURNAMENT ICM BỔ SUNG (4 chiều):
        // a. Đang ở vòng Bubble tranh Top 4 (5 người còn sống)
        f.push(if num_alive == 5 { 1.0 } else { 0.0 });
        // b. Đã lọt vào Top 4 có điểm (<= 4 người)
        f.push(if num_alive <= 4 { 1.0 } else { 0.0 });
        // c. Xếp hạng stack của mình so với những người còn sống (0.0: nghèo nhất, 1.0: giàu nhất bàn)
        let my_chips = me.stack + me.current_bet;
        let smaller_stacks_count = alive.iter().filter(|&&seat| {
            let opp_chips = self.players[seat].stack + self.players[seat].current_bet;
            my_chips > opp_chips
        }).count();
        f.push(smaller_stacks_count as f32 / (num_alive - 1).max(1) as f32);
        // d. Tỉ lệ tổng chip của mình so với toàn bộ chip trên bàn (Chip Share 0.0 .. 1.0)
        let total_chips_on_table: f32 = self.players.iter().map(|p| (p.stack + p.current_bet) as f32).sum::<f32>().max(1.0);
        f.push((my_chips as f32 / total_chips_on_table).clamp(0.0, 1.0));

        // 4. Street one-hot (4)
        let mut street_vec = [0.0f32; 4];
        street_vec[(self.street as usize).min(3)] = 1.0;
        f.extend_from_slice(&street_vec);

        // 5. Pot, to_call, stack của mình, pot odds (4)
        let to_call = self.current_highest_bet.saturating_sub(me.current_bet);
        if self.is_cash_game {
            f.push((self.pot as f32 / (bb * 150.0)).min(3.0));
            f.push((to_call as f32 / (bb * 50.0)).min(2.0));
            f.push((me.stack as f32 / (bb * 150.0)).min(2.0));
        } else {
            f.push((self.pot as f32 / (bb * 50.0)).min(2.0));
            f.push((to_call as f32 / (bb * 20.0)).min(2.0));
            f.push((me.stack as f32 / (bb * 50.0)).min(2.0));
        }
        let denom = (self.pot + to_call) as f32;
        f.push(if denom > 0.0 { to_call as f32 / denom } else { 0.0 });

        // Tiền xử lý nhanh lịch sử hành động của từng ghế trong ván bài (O(N) trên action_history)
        let mut seat_sized_bets = [0u8; MAX_PLAYERS];
        let mut seat_streets_aggr_mask = [0u8; MAX_PLAYERS];
        for r in &self.action_history {
            if r.player_idx < MAX_PLAYERS {
                match r.action {
                    Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                        seat_sized_bets[r.player_idx] = seat_sized_bets[r.player_idx].saturating_add(1);
                        seat_streets_aggr_mask[r.player_idx] |= 1 << (r.street as u8);
                    }
                    Action::AllIn => {
                        seat_streets_aggr_mask[r.player_idx] |= 1 << (r.street as u8);
                    }
                    _ => {}
                }
            }
        }

        // 6. 7 đối thủ theo thứ tự xoay tròn (7 * 4 = 28) -> index 131..158
        let mut opp_count = 0;
        let mut step = 1;
        while opp_count < (MAX_PLAYERS - 1) && step < self.num_players {
            let seat = (player_idx + step) % self.num_players;
            let o = &self.players[seat];
            if o.stack > 0 || o.is_all_in {
                // [0]: Stack còn lại & [1]: Tổng chip đối thủ đã bỏ vào pot từ đầu ván
                if self.is_cash_game {
                    f.push((o.stack as f32 / (bb * 150.0)).min(2.0));
                    f.push((o.total_contributed as f32 / (bb * 50.0)).min(2.0));
                } else {
                    f.push((o.stack as f32 / (bb * 50.0)).min(2.0));
                    f.push((o.total_contributed as f32 / (bb * 25.0)).min(2.0));
                }
                // [2]: Trạng thái trong ván + Số vòng cược đã chủ động tấn công:
                //      0.0 = đã Fold; 0.2 = chỉ Check/Call (CallStation); 0.4..1.0 = Chủ động Bet/Raise qua 1..4 vòng (TAG/Aggressor)
                let active_aggr = if o.is_in_hand() {
                    let streets_aggr = seat_streets_aggr_mask[seat].count_ones() as f32;
                    (0.2 + 0.2 * streets_aggr).min(1.0)
                } else {
                    0.0
                };
                f.push(active_aggr);
                // [3]: Cờ All-In (1.0) HOẶC Kích cỡ bet hiện tại so với Pot (0.0 .. 0.9)
                //      Giúp phân biệt rõ Maniac Shove (1.0) vs TAG Bet 33%/75% (0.25..0.50)
                let bet_profile = if o.is_all_in {
                    1.0
                } else if o.current_bet > 0 && self.pot > 0 {
                    (o.current_bet as f32 / self.pot as f32).min(0.90)
                } else {
                    0.0
                };
                f.push(bet_profile);
                opp_count += 1;
            }
            step += 1;
        }

        while opp_count < (MAX_PLAYERS - 1) {
            f.extend_from_slice(&[0.0; 4]);
            opp_count += 1;
        }

        // 7. Tóm tắt lịch sử (2) -> index 159..160
        // Chiều 159: Tổng số cú Bet/Raise theo size chuẩn (15%/33%/75%/100%) của các đối thủ trong cả ván
        let total_opp_sized_bets: u8 = (0..self.num_players)
            .filter(|&i| i != player_idx)
            .map(|i| seat_sized_bets[i])
            .sum();
        let street_raises = self
            .action_history
            .iter()
            .filter(|r| {
                r.street == self.street
                    && matches!(
                        r.action,
                        Action::BetSmall | Action::BetPot33 | Action::BetPot75 | Action::AllIn
                    )
            })
            .count();
        // Kết hợp cả số lần raise ở vòng hiện tại và tổng số lần sized-bet cả ván
        f.push(((street_raises as f32 * 0.20) + (total_opp_sized_bets as f32 * 0.20)).min(1.0));

        let in_hand = self.players.iter().filter(|p| p.is_in_hand()).count();
        f.push(in_hand as f32 / MAX_PLAYERS as f32); // index 160: F_IN_HAND

        // 8. Ngữ cảnh push/fold & sức mạnh bài (7) -> index 161..167
        let eff_bb = self.effective_stack_bb(player_idx);
        if self.is_cash_game {
            f.push((eff_bb / 150.0).min(2.0)); // index 161: dải mượt 0..300 BB
        } else {
            f.push((eff_bb / 50.0).min(1.5));  // index 161: Tournament cũ
        }
        f.push(if eff_bb < 12.0 { 1.0 } else { 0.0 }); // index 162: F_PUSHFOLD
        f.push(if to_call > 0 { 1.0 } else { 0.0 }); // index 163: F_FACING_BET
        let my_stack = me.stack as f32;
        f.push(if my_stack > 0.0 { (to_call as f32 / my_stack).min(1.0) } else { 1.0 }); // index 164
        // index 165: Tỷ lệ Pot-Commitment của bản thân (đã bỏ bao nhiêu % tổng tài sản vào pot này)
        let my_total_wealth = (me.stack + me.total_contributed).max(1) as f32;
        f.push((me.total_contributed as f32 / my_total_wealth).clamp(0.0, 1.0));
        let (cat, strength) = self.hand_strength(player_idx);
        f.push(cat);      // index 166
        f.push(strength); // index 167: F_HAND_STRENGTH

        // 9. Equity & draw & board texture (8) -> index 168..175
        f.extend_from_slice(&self.equity_features(player_idx, to_call));

        debug_assert_eq!(f.len(), FEATURE_DIM, "encode_features length mismatch");
        f
    }
}

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

use crate::environment::action::{Action, ActionMask};
use crate::environment::state::{ActionRecord, GameState, PlayerState, Street};
use crate::poker_core::deck::Deck;
use crate::poker_core::evaluator::{evaluate_7_cards, HandRank};
use rand::Rng;

/// Các action đặt cược theo % pot (pot SAU KHI call, đúng luật pot-sized raise)
const RAISE_ACTIONS: [(Action, f32); 3] = [
    (Action::BetPot33, 0.33),
    (Action::BetPot66, 0.66),
    (Action::BetPot100, 1.0),
];

#[derive(Debug, Clone)]
pub struct Environment {
    pub state: GameState,
    pub deck: Deck,
    pub street_has_acted: Vec<bool>,
}

impl Environment {
    pub fn new(num_players: usize, initial_stacks: &[u32], bb_size: u32, sb_size: u32, button_idx: usize) -> Self {
        assert!(num_players >= 2 && num_players <= 8);
        assert_eq!(initial_stacks.len(), num_players);

        let players: Vec<PlayerState> = initial_stacks
            .iter()
            .enumerate()
            .map(|(i, &stack)| PlayerState::new(i, stack, true))
            .collect();

        let state = GameState {
            num_players,
            players,
            community_cards: Vec::with_capacity(5),
            pot: 0,
            street: Street::Preflop,
            button_idx,
            current_player_idx: 0,
            min_raise: bb_size,
            current_highest_bet: 0,
            bb_size,
            sb_size,
            action_history: Vec::new(),
        };

        Environment {
            state,
            deck: Deck::new(),
            street_has_acted: vec![false; num_players],
        }
    }

    /// Reset and start a new hand
    pub fn reset_hand<R: Rng>(&mut self, rng: &mut R, button_idx: usize) {
        self.deck.reset_and_shuffle(rng);
        self.state.community_cards.clear();
        self.state.pot = 0;
        self.state.street = Street::Preflop;
        self.state.button_idx = button_idx;
        self.state.min_raise = self.state.bb_size;
        self.state.current_highest_bet = 0;
        self.state.action_history.clear();

        for p in &mut self.state.players {
            p.current_bet = 0;
            p.total_contributed = 0;
            if p.stack > 0 {
                p.is_folded = false;
                p.is_all_in = false;
                p.hole_cards = [self.deck.deal(), self.deck.deal()];
            } else {
                p.is_folded = true;
                p.is_all_in = false;
                p.hole_cards = [None, None];
            }
        }

        let surviving: Vec<usize> = (0..self.state.num_players)
            .filter(|&i| self.state.players[i].stack > 0)
            .collect();

        if surviving.len() < 2 {
            // Giải đấu kết thúc (pot = 0 nên không có chip nào cần trao)
            self.state.street = Street::Showdown;
            self.reset_street_flags();
            return;
        }

        // Map button to closest surviving player
        let num_surv = surviving.len();
        let btn_s_idx = surviving.iter().position(|&s| s >= button_idx).unwrap_or(0);
        let actual_btn = surviving[btn_s_idx];
        self.state.button_idx = actual_btn;

        let (sb_idx, bb_idx, first_actor) = if num_surv == 2 {
            // Heads-up: BTN is SB and acts first preflop
            let sb = actual_btn;
            let bb = surviving[(btn_s_idx + 1) % 2];
            (sb, bb, sb)
        } else {
            let sb = surviving[(btn_s_idx + 1) % num_surv];
            let bb = surviving[(btn_s_idx + 2) % num_surv];
            let utg = surviving[(btn_s_idx + 3) % num_surv];
            (sb, bb, utg)
        };

        self.post_blind(sb_idx, self.state.sb_size);
        self.post_blind(bb_idx, self.state.bb_size);

        // Mức bet cao nhất thực tế (BB có thể post thiếu nếu all-in vì blind)
        let highest = self.state.players.iter().map(|p| p.current_bet).max().unwrap_or(0);
        self.state.current_highest_bet = highest;
        self.reset_street_flags();

        // ✅ FIX: first_actor có thể đã all-in vì blind -> tìm người active đầu tiên
        let n = self.state.num_players;
        let actor = (0..n)
            .map(|k| (first_actor + k) % n)
            .find(|&i| self.state.players[i].is_active());

        let active_cnt = self.state.players.iter().filter(|p| p.is_active()).count();
        let needs_action = self
            .state
            .players
            .iter()
            .any(|p| p.is_active() && p.current_bet < highest);

        // Không ai có thể (hoặc cần) hành động -> chạy hết bài và chia pot ngay
        if actor.is_none() || active_cnt == 0 || (active_cnt == 1 && !needs_action) {
            self.run_out_and_showdown();
            return;
        }

        self.state.current_player_idx = actor.unwrap();
    }

    fn post_blind(&mut self, player_idx: usize, amount: u32) {
        let p = &mut self.state.players[player_idx];
        let actual = p.stack.min(amount);
        p.stack -= actual;
        p.current_bet += actual;
        p.total_contributed += actual;
        self.state.pot += actual;
        if p.stack == 0 {
            p.is_all_in = true;
        }
    }

    fn reset_street_flags(&mut self) {
        self.street_has_acted = vec![false; self.state.num_players];
    }

    /// Tổng bet mục tiêu khi raise theo `frac` pot (pot sau khi call).
    fn raise_target(&self, p_idx: usize, frac: f32) -> u32 {
        let st = &self.state;
        let cb = st.players[p_idx].current_bet;
        let to_call = st.current_highest_bet.saturating_sub(cb);
        let pot_after_call = st.pot + to_call;
        let raise_by = ((pot_after_call as f32) * frac).round() as u32;
        st.current_highest_bet + raise_by.max(st.min_raise)
    }

    /// Compute valid ActionMask for current player
    pub fn get_action_mask(&self) -> ActionMask {
        let mut mask = ActionMask::EMPTY;
        let p_idx = self.state.current_player_idx;
        let player = &self.state.players[p_idx];

        if !player.is_active() {
            return mask;
        }

        let to_call = self.state.current_highest_bet.saturating_sub(player.current_bet);
        let eff_stack_bb = self.state.effective_stack_bb(p_idx);

        if to_call > 0 {
            mask.enable(Action::Fold);
        }
        mask.enable(Action::CheckCall);

        if player.stack > to_call {
            mask.enable(Action::AllIn);
        }

        // PUSH / FOLD MODE: effective stack < 12 BB -> chỉ [Fold, CheckCall, AllIn]
        if eff_stack_bb < 12.0 {
            return mask;
        }

        if player.stack > to_call {
            let min_raise_total = self.state.current_highest_bet + self.state.min_raise;
            let mut last_target = 0u32;
            for &(act, frac) in RAISE_ACTIONS.iter() {
                let target = self.raise_target(p_idx, frac);
                let needed = target.saturating_sub(player.current_bet);
                // ✅ FIX: bỏ action trùng mức bet (33%/66%/100% trùng nhau khi pot nhỏ)
                if needed < player.stack && target >= min_raise_total && target > last_target {
                    mask.enable(act);
                    last_target = target;
                }
            }
        }

        mask
    }

    /// Đẩy toàn bộ stack. Trả về số chip đã đưa vào.
    fn commit_all_in(&mut self, p_idx: usize) -> u32 {
        let all_in_amt = self.state.players[p_idx].stack;
        let new_total = self.state.players[p_idx].current_bet + all_in_amt;
        self.state.players[p_idx].stack = 0;
        self.state.players[p_idx].is_all_in = true;
        self.state.players[p_idx].current_bet = new_total;
        self.state.players[p_idx].total_contributed += all_in_amt;
        self.state.pot += all_in_amt;

        if new_total > self.state.current_highest_bet {
            let raise_size = new_total - self.state.current_highest_bet;
            // ✅ FIX: chỉ mở lại action nếu là full raise. Short all-in không reset.
            // (người đã hành động vẫn phải call phần chênh lệch vì current_bet < highest)
            if raise_size >= self.state.min_raise {
                self.state.min_raise = raise_size;
                self.street_has_acted.fill(false);
            }
            self.state.current_highest_bet = new_total;
        }
        all_in_amt
    }

    /// Execute action and advance game state
    pub fn step(&mut self, action: Action) -> Result<bool, &'static str> {
        let mask = self.get_action_mask();
        if !mask.is_valid(action) {
            return Err("Illegal action attempted according to action mask!");
        }

        let p_idx = self.state.current_player_idx;
        let to_call = self
            .state
            .current_highest_bet
            .saturating_sub(self.state.players[p_idx].current_bet);

        let committed: u32;

        match action {
            Action::Fold => {
                self.state.players[p_idx].is_folded = true;
                committed = 0;
            }
            Action::CheckCall => {
                let call_amount = self.state.players[p_idx].stack.min(to_call);
                self.state.players[p_idx].stack -= call_amount;
                self.state.players[p_idx].current_bet += call_amount;
                self.state.players[p_idx].total_contributed += call_amount;
                self.state.pot += call_amount;
                committed = call_amount;
                if self.state.players[p_idx].stack == 0 {
                    self.state.players[p_idx].is_all_in = true;
                }
            }
            Action::BetPot33 | Action::BetPot66 | Action::BetPot100 => {
                let frac = match action {
                    Action::BetPot33 => 0.33f32,
                    Action::BetPot66 => 0.66f32,
                    _ => 1.0f32,
                };
                let target_total = self.raise_target(p_idx, frac);
                let needed = target_total.saturating_sub(self.state.players[p_idx].current_bet);

                if needed >= self.state.players[p_idx].stack {
                    committed = self.commit_all_in(p_idx);
                } else {
                    self.state.players[p_idx].stack -= needed;
                    self.state.players[p_idx].current_bet = target_total;
                    self.state.players[p_idx].total_contributed += needed;
                    self.state.pot += needed;
                    committed = needed;

                    let raise_size = target_total.saturating_sub(self.state.current_highest_bet);
                    if raise_size > self.state.min_raise {
                        self.state.min_raise = raise_size;
                    }
                    self.state.current_highest_bet = target_total;
                    self.street_has_acted.fill(false);
                }
            }
            Action::AllIn => {
                committed = self.commit_all_in(p_idx);
            }
        }

        self.street_has_acted[p_idx] = true;
        self.state.action_history.push(ActionRecord {
            player_idx: p_idx,
            street: self.state.street,
            action,
            amount: committed,
        });

        // Hand terminal nếu chỉ còn 1 người chưa fold
        let active_in_hand: Vec<usize> = self
            .state
            .players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_in_hand())
            .map(|(i, _)| i)
            .collect();

        if active_in_hand.len() <= 1 {
            self.resolve_terminal_fold(active_in_hand.first().copied());
            return Ok(true);
        }

        if self.is_betting_round_complete() {
            let hand_ended = self.advance_street();
            return Ok(hand_ended);
        }

        self.advance_to_next_player();
        Ok(false)
    }

    fn is_betting_round_complete(&self) -> bool {
        for (i, p) in self.state.players.iter().enumerate() {
            if p.is_active() && (!self.street_has_acted[i] || p.current_bet < self.state.current_highest_bet) {
                return false;
            }
        }
        true
    }

    fn advance_to_next_player(&mut self) {
        let num_p = self.state.num_players;
        let mut next = (self.state.current_player_idx + 1) % num_p;
        let start = next;
        loop {
            if self.state.players[next].is_active() {
                self.state.current_player_idx = next;
                return;
            }
            next = (next + 1) % num_p;
            if next == start {
                break;
            }
        }
    }

    /// Chia nốt các lá còn thiếu, sang Showdown và chia pot.
    fn run_out_and_showdown(&mut self) {
        while self.state.community_cards.len() < 5 {
            match self.deck.deal() {
                Some(c) => self.state.community_cards.push(c),
                None => break,
            }
        }
        self.state.street = Street::Showdown;
        self.resolve_showdown();
    }

    /// Advance to next street (Flop -> Turn -> River -> Showdown). Trả về true nếu hand kết thúc.
    fn advance_street(&mut self) -> bool {
        for p in &mut self.state.players {
            p.current_bet = 0;
        }
        self.state.current_highest_bet = 0;
        self.state.min_raise = self.state.bb_size;
        self.reset_street_flags();

        let can_act_count = self.state.players.iter().filter(|p| p.is_active()).count();

        match self.state.street {
            Street::Preflop => {
                self.state.street = Street::Flop;
                for _ in 0..3 {
                    if let Some(c) = self.deck.deal() {
                        self.state.community_cards.push(c);
                    }
                }
            }
            Street::Flop => {
                self.state.street = Street::Turn;
                if let Some(c) = self.deck.deal() {
                    self.state.community_cards.push(c);
                }
            }
            Street::Turn => {
                self.state.street = Street::River;
                if let Some(c) = self.deck.deal() {
                    self.state.community_cards.push(c);
                }
            }
            Street::River => {
                self.state.street = Street::Showdown;
                self.resolve_showdown();
                return true;
            }
            Street::Showdown => return true,
        }

        // <= 1 người còn có thể hành động -> chạy hết bài tới showdown
        if can_act_count <= 1 {
            self.run_out_and_showdown();
            return true;
        }

        // Người đầu tiên hành động post-flop: active đầu tiên sau BTN
        let num_p = self.state.num_players;
        let mut next = (self.state.button_idx + 1) % num_p;
        while !self.state.players[next].is_active() {
            next = (next + 1) % num_p;
        }
        self.state.current_player_idx = next;

        false
    }

    /// Single player left after all others folded
    fn resolve_terminal_fold(&mut self, winner_opt: Option<usize>) {
        self.state.street = Street::Showdown;
        if let Some(winner_idx) = winner_opt {
            self.state.players[winner_idx].stack += self.state.pot;
            self.state.pot = 0;
        }
    }

    /// Chia `amount` chip cho người thắng trong `eligible` (chia đều, lẻ cho người đầu tiên)
    fn award(&mut self, eligible: &[usize], amount: u32, scores: &[(usize, HandRank)]) {
        let best = eligible
            .iter()
            .filter_map(|&i| scores.iter().find(|&&(s, _)| s == i).map(|&(_, r)| r))
            .max();
        let Some(best) = best else { return };

        let winners: Vec<usize> = eligible
            .iter()
            .copied()
            .filter(|&i| scores.iter().any(|&(s, r)| s == i && r == best))
            .collect();
        if winners.is_empty() {
            return;
        }

        let share = amount / (winners.len() as u32);
        let mut remainder = amount % (winners.len() as u32);
        for &w in &winners {
            let bonus = if remainder > 0 {
                remainder -= 1;
                1
            } else {
                0
            };
            self.state.players[w].stack += share + bonus;
        }
    }

    /// Compute side pots and distribute winnings at showdown
    pub fn resolve_showdown(&mut self) {
        let mut scores: Vec<(usize, HandRank)> = Vec::new();
        for (i, p) in self.state.players.iter().enumerate() {
            if p.is_in_hand() {
                let mut cards = self.state.community_cards.clone();
                if let Some(c0) = p.hole_cards[0] {
                    cards.push(c0);
                }
                if let Some(c1) = p.hole_cards[1] {
                    cards.push(c1);
                }
                if cards.len() >= 5 {
                    scores.push((i, evaluate_7_cards(&cards)));
                }
            }
        }

        if scores.is_empty() {
            return;
        }

        let mut contributions: Vec<(usize, u32)> = self
            .state
            .players
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.total_contributed))
            .filter(|&(_, amt)| amt > 0)
            .collect();

        while !contributions.is_empty() && self.state.pot > 0 {
            let min_contrib = contributions.iter().map(|&(_, amt)| amt).min().unwrap();
            let slice = (min_contrib * (contributions.len() as u32)).min(self.state.pot);

            let eligible: Vec<usize> = contributions
                .iter()
                .map(|&(idx, _)| idx)
                .filter(|&idx| scores.iter().any(|&(s, _)| s == idx))
                .collect();

            // ✅ FIX: chỉ trừ pot khi thực sự có người nhận (bảo toàn chip)
            if !eligible.is_empty() {
                self.award(&eligible, slice, &scores);
                self.state.pot -= slice;
            }

            for (_, amt) in &mut contributions {
                *amt -= min_contrib;
            }
            contributions.retain(|&(_, amt)| amt > 0);
        }

        // Phần pot còn dư (lát không ai đủ điều kiện) -> người thắng chung cuộc
        if self.state.pot > 0 {
            let all: Vec<usize> = scores.iter().map(|&(i, _)| i).collect();
            let pot = self.state.pot;
            self.award(&all, pot, &scores);
            self.state.pot = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::mock::StepRng;

    #[test]
    fn test_push_fold_activation_under_12bb() {
        let mut rng = StepRng::new(42, 1);
        let stacks = vec![1000, 1000]; // 10 BB (BB = 100)
        let mut env = Environment::new(2, &stacks, 100, 50, 0);
        env.reset_hand(&mut rng, 0);

        let mask = env.get_action_mask();
        assert!(!mask.is_valid(Action::BetPot33));
        assert!(!mask.is_valid(Action::BetPot66));
        assert!(!mask.is_valid(Action::BetPot100));
        assert!(mask.is_valid(Action::AllIn));
    }

    #[test]
    fn test_chip_conservation_random_play() {
        let mut rng = rand::thread_rng();
        for _ in 0..300 {
            let n = rng.gen_range(2..=8);
            // Stack nhỏ (50 = bằng SB) để kiểm tra trường hợp all-in ngay từ blind
            let stacks: Vec<u32> = (0..n).map(|_| rng.gen_range(1..=60) * 50).collect();
            let total: u32 = stacks.iter().sum();
            let mut env = Environment::new(n, &stacks, 100, 50, 0);
            env.reset_hand(&mut rng, 0);

            let mut steps = 0;
            while env.state.street != Street::Showdown && steps < 500 {
                let valid = env.get_action_mask().valid_actions();
                assert!(!valid.is_empty(), "current player has no legal action (deadlock)");
                let a = valid[rng.gen_range(0..valid.len())];
                env.step(a).unwrap();
                steps += 1;
            }
            assert!(env.state.street == Street::Showdown, "hand did not finish");
            let after: u32 = env.state.players.iter().map(|p| p.stack).sum::<u32>() + env.state.pot;
            assert_eq!(total, after, "chips not conserved");
            assert_eq!(env.state.pot, 0);
        }
    }
}

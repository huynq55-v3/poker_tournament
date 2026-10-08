use crate::environment::action::{Action, ActionMask};
use crate::environment::state::{GameState, PlayerState, Street, ActionRecord};
use crate::poker_core::deck::Deck;
use crate::poker_core::evaluator::{evaluate_7_cards, HandRank};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Environment {
    pub state: GameState,
    pub deck: Deck,
    pub active_players_in_street: usize,
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
            active_players_in_street: num_players,
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

        // Reset players
        for p in &mut self.state.players {
            p.current_bet = 0;
            p.total_contributed = 0;
            p.is_folded = p.stack == 0;
            p.is_all_in = false;
            p.hole_cards = [self.deck.deal(), self.deck.deal()];
        }

        // Post Blinds
        let num_p = self.state.num_players;
        let (sb_idx, bb_idx, first_actor) = if num_p == 2 {
            // Heads up: BTN posts SB and acts first preflop, other posts BB
            (button_idx, (button_idx + 1) % num_p, button_idx)
        } else {
            let sb = (button_idx + 1) % num_p;
            let bb = (button_idx + 2) % num_p;
            let utg = (button_idx + 3) % num_p;
            (sb, bb, utg)
        };

        self.post_blind(sb_idx, self.state.sb_size);
        self.post_blind(bb_idx, self.state.bb_size);

        self.state.current_highest_bet = self.state.bb_size;
        self.state.current_player_idx = first_actor;
        self.reset_street_flags();
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

        // 1. Fold is always valid if facing a bet (to_call > 0). If to_call == 0, folding is technically inferior to check,
        // but allowed in some models. Standard GTO practice: if to_call > 0, Fold is valid; if to_call == 0, CheckCall is free.
        if to_call > 0 {
            mask.enable(Action::Fold);
        }

        // 2. Check / Call is always valid
        mask.enable(Action::CheckCall);

        // 3. All-in is valid if player has more chips than to_call
        if player.stack > to_call {
            mask.enable(Action::AllIn);
        }

        // PUSH / FOLD MODE:
        // If effective stack < 12 BB, collapse betting actions to strictly [Fold, CheckCall, AllIn]!
        if eff_stack_bb < 12.0 {
            return mask;
        }

        // 4. Pot Sized Bets / Raises (33%, 66%, 100%)
        if player.stack > to_call {
            let pot = self.state.pot;
            let min_raise_total = self.state.current_highest_bet + self.state.min_raise;

            // Bet 33%
            let add_33 = ((pot as f32) * 0.33).round() as u32;
            let target_bet_33 = self.state.current_highest_bet + add_33.max(self.state.min_raise);
            let needed_33 = target_bet_33.saturating_sub(player.current_bet);
            if needed_33 < player.stack && target_bet_33 >= min_raise_total {
                mask.enable(Action::BetPot33);
            }

            // Bet 66%
            let add_66 = ((pot as f32) * 0.66).round() as u32;
            let target_bet_66 = self.state.current_highest_bet + add_66.max(self.state.min_raise);
            let needed_66 = target_bet_66.saturating_sub(player.current_bet);
            if needed_66 < player.stack && target_bet_66 >= min_raise_total {
                mask.enable(Action::BetPot66);
            }

            // Bet 100%
            let add_100 = pot;
            let target_bet_100 = self.state.current_highest_bet + add_100.max(self.state.min_raise);
            let needed_100 = target_bet_100.saturating_sub(player.current_bet);
            if needed_100 < player.stack && target_bet_100 >= min_raise_total {
                mask.enable(Action::BetPot100);
            }
        }

        mask
    }

    /// Execute action and advance game state
    pub fn step(&mut self, action: Action) -> Result<bool, &'static str> {
        let mask = self.get_action_mask();
        if !mask.is_valid(action) {
            return Err("Illegal action attempted according to action mask!");
        }

        let p_idx = self.state.current_player_idx;
        let to_call = self.state.current_highest_bet.saturating_sub(self.state.players[p_idx].current_bet);
        let pot = self.state.pot;

        let mut committed = 0u32;

        match action {
            Action::Fold => {
                self.state.players[p_idx].is_folded = true;
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
                let add = ((pot as f32) * frac).round() as u32;
                let mut target_total = self.state.current_highest_bet + add.max(self.state.min_raise);
                let needed = target_total.saturating_sub(self.state.players[p_idx].current_bet);

                if needed >= self.state.players[p_idx].stack {
                    // Falls into AllIn
                    let all_in_amt = self.state.players[p_idx].stack;
                    target_total = self.state.players[p_idx].current_bet + all_in_amt;
                    self.state.players[p_idx].stack = 0;
                    self.state.players[p_idx].is_all_in = true;
                    self.state.players[p_idx].current_bet = target_total;
                    self.state.players[p_idx].total_contributed += all_in_amt;
                    self.state.pot += all_in_amt;
                    committed = all_in_amt;
                } else {
                    self.state.players[p_idx].stack -= needed;
                    self.state.players[p_idx].current_bet = target_total;
                    self.state.players[p_idx].total_contributed += needed;
                    self.state.pot += needed;
                    committed = needed;
                }

                let raise_size = target_total.saturating_sub(self.state.current_highest_bet);
                if raise_size > self.state.min_raise {
                    self.state.min_raise = raise_size;
                }
                self.state.current_highest_bet = target_total;

                // When someone raises, other active players must react
                self.street_has_acted.fill(false);
            }
            Action::AllIn => {
                let all_in_amt = self.state.players[p_idx].stack;
                let new_total = self.state.players[p_idx].current_bet + all_in_amt;
                self.state.players[p_idx].stack = 0;
                self.state.players[p_idx].is_all_in = true;
                self.state.players[p_idx].current_bet = new_total;
                self.state.players[p_idx].total_contributed += all_in_amt;
                self.state.pot += all_in_amt;
                committed = all_in_amt;

                if new_total > self.state.current_highest_bet {
                    let raise_size = new_total - self.state.current_highest_bet;
                    if raise_size >= self.state.min_raise {
                        self.state.min_raise = raise_size;
                    }
                    self.state.current_highest_bet = new_total;
                    self.street_has_acted.fill(false);
                }
            }
        }

        self.street_has_acted[p_idx] = true;
        self.state.action_history.push(ActionRecord {
            player_idx: p_idx,
            street: self.state.street,
            action,
            amount: committed,
        });

        // Check if hand is terminal (only 1 player remains unfolded)
        let active_in_hand: Vec<usize> = self.state.players.iter()
            .enumerate()
            .filter(|(_, p)| p.is_in_hand())
            .map(|(i, _)| i)
            .collect();

        if active_in_hand.len() <= 1 {
            self.resolve_terminal_fold(active_in_hand.first().copied());
            return Ok(true); // Hand completed
        }

        // Check if betting round for this street is complete
        if self.is_betting_round_complete() {
            let hand_ended = self.advance_street();
            return Ok(hand_ended);
        }

        // Move to next active player
        self.advance_to_next_player();
        Ok(false)
    }

    fn is_betting_round_complete(&self) -> bool {
        for (i, p) in self.state.players.iter().enumerate() {
            if p.is_active() {
                if !self.street_has_acted[i] || p.current_bet < self.state.current_highest_bet {
                    return false;
                }
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

    /// Advance to next street (Flop -> Turn -> River -> Showdown)
    fn advance_street(&mut self) -> bool {
        // Reset current_bet for next street
        for p in &mut self.state.players {
            p.current_bet = 0;
        }
        self.state.current_highest_bet = 0;
        self.state.min_raise = self.state.bb_size;
        self.reset_street_flags();

        // Check how many players can still act
        let can_act_count = self.state.players.iter().filter(|p| p.is_active()).count();

        match self.state.street {
            Street::Preflop => {
                self.state.street = Street::Flop;
                // Deal 3 Flop cards
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
            Street::Showdown => {
                return true;
            }
        }

        // If <= 1 players can still act (others all-in or folded), run out the board directly to showdown!
        if can_act_count <= 1 {
            while self.state.community_cards.len() < 5 {
                if let Some(c) = self.deck.deal() {
                    self.state.community_cards.push(c);
                }
            }
            self.state.street = Street::Showdown;
            self.resolve_showdown();
            return true;
        }

        // Find first active player post-flop (starts from first active seat after BTN)
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

    /// Compute side pots and distribute winnings at showdown
    pub fn resolve_showdown(&mut self) {
        // Collect evaluations for non-folded players
        let mut scores: Vec<(usize, HandRank)> = Vec::new();
        for (i, p) in self.state.players.iter().enumerate() {
            if p.is_in_hand() {
                let mut cards = self.state.community_cards.clone();
                if let Some(c0) = p.hole_cards[0] { cards.push(c0); }
                if let Some(c1) = p.hole_cards[1] { cards.push(c1); }
                if cards.len() >= 5 {
                    let score = evaluate_7_cards(&cards);
                    scores.push((i, score));
                }
            }
        }

        if scores.is_empty() {
            return;
        }

        // Calculate Side Pots based on total_contributed
        let mut contributions: Vec<(usize, u32)> = self.state.players.iter()
            .enumerate()
            .map(|(i, p)| (i, p.total_contributed))
            .filter(|&(_, amt)| amt > 0)
            .collect();

        while !contributions.is_empty() && self.state.pot > 0 {
            // Find lowest contribution
            let min_contrib = contributions.iter().map(|&(_, amt)| amt).min().unwrap();
            let pot_slice = min_contrib * (contributions.len() as u32);
            let pot_to_distribute = pot_slice.min(self.state.pot);

            // Eligible players for this pot slice are non-folded contributors
            let eligible: Vec<usize> = contributions.iter()
                .map(|&(idx, _)| idx)
                .filter(|&idx| self.state.players[idx].is_in_hand())
                .collect();

            if !eligible.is_empty() {
                // Find winners among eligible
                let mut best_score: Option<HandRank> = None;
                for &idx in &eligible {
                    if let Some(&(_, score)) = scores.iter().find(|&&(s_idx, _)| s_idx == idx) {
                        match best_score {
                            None => best_score = Some(score),
                            Some(bs) => if score > bs { best_score = Some(score); }
                        }
                    }
                }

                if let Some(bs) = best_score {
                    let winners: Vec<usize> = eligible.iter()
                        .copied()
                        .filter(|&idx| {
                            scores.iter().any(|&(s_idx, s)| s_idx == idx && s == bs)
                        })
                        .collect();

                    let share = pot_to_distribute / (winners.len() as u32);
                    let mut remainder = pot_to_distribute % (winners.len() as u32);

                    for &w in &winners {
                        let bonus = if remainder > 0 { remainder -= 1; 1 } else { 0 };
                        self.state.players[w].stack += share + bonus;
                    }
                }
            }

            self.state.pot -= pot_to_distribute;

            // Subtract min_contrib from all contributors
            for (_, amt) in &mut contributions {
                *amt -= min_contrib;
            }
            contributions.retain(|&(_, amt)| amt > 0);
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
        // Since eff stack is 10 BB (< 12 BB), bet sizes (33%, 66%, 100%) must NOT be enabled!
        assert!(!mask.is_valid(Action::BetPot33));
        assert!(!mask.is_valid(Action::BetPot66));
        assert!(!mask.is_valid(Action::BetPot100));
        assert!(mask.is_valid(Action::AllIn));
    }
}

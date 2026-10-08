use crate::poker_core::card::{Card, Rank, Suit};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Deck {
    cards: [Card; 52],
    index: usize,
}

impl Deck {
    pub fn new() -> Self {
        let mut cards = [Card(0); 52];
        let mut idx = 0;
        for &rank in &Rank::ALL {
            for &suit in &Suit::ALL {
                cards[idx] = Card::new(rank, suit);
                idx += 1;
            }
        }
        Deck { cards, index: 0 }
    }

    #[inline(always)]
    pub fn reset_and_shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.index = 0;
        // Fisher-Yates in-place shuffle
        for i in (1..52).rev() {
            let j = rng.gen_range(0..=i);
            self.cards.swap(i, j);
        }
    }

    #[inline(always)]
    pub fn deal(&mut self) -> Option<Card> {
        if self.index < 52 {
            let card = self.cards[self.index];
            self.index += 1;
            Some(card)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn remaining(&self) -> usize {
        52 - self.index
    }
}

impl Default for Deck {
    fn default() -> Self {
        Self::new()
    }
}

use crate::poker_core::card::{Card, Rank};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum HandCategory {
    HighCard = 1,
    OnePair = 2,
    TwoPair = 3,
    ThreeOfAKind = 4,
    Straight = 5,
    Flush = 6,
    FullHouse = 7,
    FourOfAKind = 8,
    StraightFlush = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HandRank(pub u32);

impl HandRank {
    #[inline(always)]
    pub fn category(&self) -> HandCategory {
        match (self.0 >> 24) as u8 {
            1 => HandCategory::HighCard,
            2 => HandCategory::OnePair,
            3 => HandCategory::TwoPair,
            4 => HandCategory::ThreeOfAKind,
            5 => HandCategory::Straight,
            6 => HandCategory::Flush,
            7 => HandCategory::FullHouse,
            8 => HandCategory::FourOfAKind,
            _ => HandCategory::StraightFlush,
        }
    }

    #[inline(always)]
    pub fn score(&self) -> u32 {
        self.0
    }
}

/// Helper to pack hand score into u32:
/// (category << 24) | (r0 << 16) | (r1 << 12) | (r2 << 8) | (r3 << 4) | r4
#[inline(always)]
fn encode_score(cat: HandCategory, r0: u8, r1: u8, r2: u8, r3: u8, r4: u8) -> HandRank {
    let score = ((cat as u32) << 24)
        | ((r0 as u32) << 16)
        | ((r1 as u32) << 12)
        | ((r2 as u32) << 8)
        | ((r3 as u32) << 4)
        | (r4 as u32);
    HandRank(score)
}

/// Fast bitwise 5-to-7 card evaluator
pub fn evaluate_7_cards(cards: &[Card]) -> HandRank {
    debug_assert!(cards.len() >= 5 && cards.len() <= 7);

    let mut suit_masks = [0u16; 4];
    let mut suit_counts = [0u8; 4];
    let mut rank_counts = [0u8; 13];
    let mut ranks_mask = 0u16;

    for &card in cards {
        let r = card.rank_val() as usize;
        let s = card.suit_val() as usize;
        suit_masks[s] |= 1u16 << r;
        suit_counts[s] += 1;
        rank_counts[r] += 1;
        ranks_mask |= 1u16 << r;
    }

    // 1. Check for Flush and Straight Flush
    for s in 0..4 {
        if suit_counts[s] >= 5 {
            let smask = suit_masks[s];
            if let Some(top_straight) = check_straight_in_mask(smask) {
                return encode_score(HandCategory::StraightFlush, top_straight, 0, 0, 0, 0);
            }
            // Normal Flush: extract highest 5 ranks in suit
            let (r0, r1, r2, r3, r4) = top_5_from_mask(smask);
            return encode_score(HandCategory::Flush, r0, r1, r2, r3, r4);
        }
    }

    // 2. Check Four of a Kind
    for r in (0..13).rev() {
        if rank_counts[r] == 4 {
            let quad_rank = r as u8;
            let kicker = (0..13)
                .rev()
                .find(|&k| k != r && rank_counts[k] > 0)
                .unwrap_or(0) as u8;
            return encode_score(HandCategory::FourOfAKind, quad_rank, kicker, 0, 0, 0);
        }
    }

    // 3. Check Full House & Three of a Kind
    let mut trips_rank = None;
    let mut second_trips_or_pair = None;

    for r in (0..13).rev() {
        if rank_counts[r] >= 3 {
            if trips_rank.is_none() {
                trips_rank = Some(r as u8);
            } else if second_trips_or_pair.is_none() {
                second_trips_or_pair = Some(r as u8);
            }
        } else if rank_counts[r] >= 2 && second_trips_or_pair.is_none() {
            second_trips_or_pair = Some(r as u8);
        }
    }

    // If trips_rank is found, but we might have a pair lower than trips that was missed:
    if let Some(tr) = trips_rank {
        if second_trips_or_pair.is_none() {
            for r in (0..13).rev() {
                if r != tr as usize && rank_counts[r] >= 2 {
                    second_trips_or_pair = Some(r as u8);
                    break;
                }
            }
        }
        if let Some(pair_r) = second_trips_or_pair {
            return encode_score(HandCategory::FullHouse, tr, pair_r, 0, 0, 0);
        }
    }

    // 4. Check Straight
    if let Some(top_straight) = check_straight_in_mask(ranks_mask) {
        return encode_score(HandCategory::Straight, top_straight, 0, 0, 0, 0);
    }

    // 5. Three of a kind (already found trips_rank, but no second pair for full house)
    if let Some(tr) = trips_rank {
        let mut kickers = [0u8; 2];
        let mut k_idx = 0;
        for r in (0..13).rev() {
            if r != tr as usize && rank_counts[r] > 0 {
                kickers[k_idx] = r as u8;
                k_idx += 1;
                if k_idx == 2 {
                    break;
                }
            }
        }
        return encode_score(HandCategory::ThreeOfAKind, tr, kickers[0], kickers[1], 0, 0);
    }

    // 6. Check Two Pair & One Pair
    let mut pairs = [0u8; 3];
    let mut pair_count = 0;
    for r in (0..13).rev() {
        if rank_counts[r] >= 2 {
            pairs[pair_count] = r as u8;
            pair_count += 1;
            if pair_count == 3 {
                break;
            }
        }
    }

    if pair_count >= 2 {
        let p1 = pairs[0];
        let p2 = pairs[1];
        let kicker = (0..13)
            .rev()
            .find(|&r| r != p1 as usize && r != p2 as usize && rank_counts[r] > 0)
            .unwrap_or(0) as u8;
        return encode_score(HandCategory::TwoPair, p1, p2, kicker, 0, 0);
    } else if pair_count == 1 {
        let p1 = pairs[0];
        let mut kickers = [0u8; 3];
        let mut k_idx = 0;
        for r in (0..13).rev() {
            if r != p1 as usize && rank_counts[r] > 0 {
                kickers[k_idx] = r as u8;
                k_idx += 1;
                if k_idx == 3 {
                    break;
                }
            }
        }
        return encode_score(HandCategory::OnePair, p1, kickers[0], kickers[1], kickers[2], 0);
    }

    // 7. High Card
    let (r0, r1, r2, r3, r4) = top_5_from_mask(ranks_mask);
    encode_score(HandCategory::HighCard, r0, r1, r2, r3, r4)
}

#[inline(always)]
fn check_straight_in_mask(mask: u16) -> Option<u8> {
    // Check 5-consecutive bits from Ace down to 5
    // i = 8: ranks 8..12 (Ten to Ace) -> Broadway Straight
    for i in (0..=8).rev() {
        let pattern = 0b11111 << i;
        if (mask & pattern) == pattern {
            return Some((i + 4) as u8); // Top card rank
        }
    }
    // Check Ace-low wheel: A-2-3-4-5 -> mask contains bits 12 (Ace) and 0, 1, 2, 3 (2, 3, 4, 5)
    // 0x100F = 0b0001_0000_0000_1111
    if (mask & 0x100F) == 0x100F {
        return Some(Rank::Five as u8); // Top card is 5
    }
    None
}

#[inline(always)]
fn top_5_from_mask(mask: u16) -> (u8, u8, u8, u8, u8) {
    let mut top = [0u8; 5];
    let mut count = 0;
    for r in (0..13).rev() {
        if (mask & (1 << r)) != 0 {
            top[count] = r as u8;
            count += 1;
            if count == 5 {
                break;
            }
        }
    }
    (top[0], top[1], top[2], top[3], top[4])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_royal_flush_vs_straight_flush() {
        // Royal Flush: As Ks Qs Js Ts 2c 3d
        let rf = vec![
            Card::from_str_repr("As").unwrap(),
            Card::from_str_repr("Ks").unwrap(),
            Card::from_str_repr("Qs").unwrap(),
            Card::from_str_repr("Js").unwrap(),
            Card::from_str_repr("Ts").unwrap(),
            Card::from_str_repr("2c").unwrap(),
            Card::from_str_repr("3d").unwrap(),
        ];
        // King-high Straight Flush: Ks Qs Js Ts 9s 2c 3d
        let sf = vec![
            Card::from_str_repr("Ks").unwrap(),
            Card::from_str_repr("Qs").unwrap(),
            Card::from_str_repr("Js").unwrap(),
            Card::from_str_repr("Ts").unwrap(),
            Card::from_str_repr("9s").unwrap(),
            Card::from_str_repr("2c").unwrap(),
            Card::from_str_repr("3d").unwrap(),
        ];
        let score_rf = evaluate_7_cards(&rf);
        let score_sf = evaluate_7_cards(&sf);

        assert_eq!(score_rf.category(), HandCategory::StraightFlush);
        assert_eq!(score_sf.category(), HandCategory::StraightFlush);
        assert!(score_rf > score_sf);
    }

    #[test]
    fn test_wheel_straight_flush() {
        // As 2s 3s 4s 5s
        let wheel = vec![
            Card::from_str_repr("As").unwrap(),
            Card::from_str_repr("2s").unwrap(),
            Card::from_str_repr("3s").unwrap(),
            Card::from_str_repr("4s").unwrap(),
            Card::from_str_repr("5s").unwrap(),
            Card::from_str_repr("9c").unwrap(),
            Card::from_str_repr("Kd").unwrap(),
        ];
        let score = evaluate_7_cards(&wheel);
        assert_eq!(score.category(), HandCategory::StraightFlush);
        // Quads: 9s 9h 9d 9c Ad Kd Qd
        let quads = vec![
            Card::from_str_repr("9s").unwrap(),
            Card::from_str_repr("9h").unwrap(),
            Card::from_str_repr("9d").unwrap(),
            Card::from_str_repr("9c").unwrap(),
            Card::from_str_repr("Ad").unwrap(),
            Card::from_str_repr("Kd").unwrap(),
            Card::from_str_repr("Qd").unwrap(),
        ];
        let score_quads = evaluate_7_cards(&quads);
        assert!(score > score_quads);
    }
}

//! Equity so với bài ngẫu nhiên của đối thủ + đặc trưng draw/board.
//! - Preflop: bảng tính sẵn 169 lớp bài x 1..=7 đối thủ (tự build lần đầu, lưu preflop_equity.json)
//! - Postflop: Monte Carlo với RNG xác định theo bài (cùng tình huống -> cùng kết quả, GUI không bị nhấp nháy)

use crate::poker_core::card::{Card, Rank, Suit};
use crate::poker_core::evaluator::{evaluate_7_cards, HandRank};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub const PREFLOP_CLASSES: usize = 169;
pub const MAX_OPP: usize = 7;
pub const PREFLOP_TABLE_PATH: &str = "preflop_equity.json";
const PREFLOP_BUILD_SAMPLES: usize = 8_000;
/// Số mẫu Monte Carlo postflop. 64 ~ sai số ±6%; tăng lên 128+ nếu muốn chính xác hơn (chậm hơn).
pub const POSTFLOP_SAMPLES: usize = 64;

// ---------------------------------------------------------------- RNG nhỏ, không phụ thuộc crate

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        let s = splitmix64(seed);
        XorShift(if s == 0 { 0x1234_5678_9ABC_DEF1 } else { s })
    }
    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline(always)]
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Seed không phụ thuộc thứ tự lá bài (hole và board được cộng giao hoán)
fn seed_for(hole: [Card; 2], board: &[Card], num_opp: usize) -> u64 {
    let mut s = 0u64;
    for c in hole {
        s = s.wrapping_add(splitmix64(c.0 as u64 + 1));
    }
    for c in board {
        s = s.wrapping_add(splitmix64(c.0 as u64 + 101));
    }
    s ^ splitmix64(num_opp as u64 + 7777)
}

// ---------------------------------------------------------------- Monte Carlo equity

/// Equity (0..1) của `hole` so với `num_opp` đối thủ có bài ngẫu nhiên. Hòa chia đều.
pub fn monte_carlo_equity(
    hole: [Card; 2],
    board: &[Card],
    num_opp: usize,
    samples: usize,
    rng: &mut XorShift,
) -> f32 {
    if num_opp == 0 {
        return 1.0;
    }
    let mut dead = [false; 52];
    for c in hole.iter().chain(board.iter()) {
        dead[c.0 as usize] = true;
    }
    let mut pool = [0u8; 52];
    let mut m = 0usize;
    for i in 0..52u8 {
        if !dead[i as usize] {
            pool[m] = i;
            m += 1;
        }
    }

    let board_len = board.len().min(5);
    let need_board = 5 - board_len;
    let k = need_board + 2 * num_opp;
    if m < k || samples == 0 {
        return 1.0 / (num_opp as f32 + 1.0);
    }

    let mut full = [Card(0); 7];
    for (i, c) in board.iter().take(5).enumerate() {
        full[i] = *c;
    }

    let mut total = 0.0f32;
    for _ in 0..samples {
        // Partial Fisher-Yates: chỉ xáo k lá đầu
        for i in 0..k {
            let j = i + rng.below(m - i);
            pool.swap(i, j);
        }
        for i in 0..need_board {
            full[board_len + i] = Card(pool[i]);
        }

        full[5] = hole[0];
        full[6] = hole[1];
        let my = evaluate_7_cards(&full);

        let mut best: Option<HandRank> = None;
        let mut ties = 0u32;
        for o in 0..num_opp {
            full[5] = Card(pool[need_board + 2 * o]);
            full[6] = Card(pool[need_board + 2 * o + 1]);
            let r = evaluate_7_cards(&full);
            match best {
                Some(b) if r < b => {}
                Some(b) if r == b => ties += 1,
                _ => {
                    best = Some(r);
                    ties = 1;
                }
            }
        }
        let b = best.unwrap();
        total += if my > b {
            1.0
        } else if my == b {
            1.0 / (ties as f32 + 1.0)
        } else {
            0.0
        };
    }
    total / samples as f32
}

// ---------------------------------------------------------------- Bảng preflop

/// Chỉ số lớp bài 0..169: 13 đôi + 78 suited + 78 offsuit
pub fn class_index(c0: Card, c1: Card) -> usize {
    let (a, b) = (c0.rank_val() as usize, c1.rank_val() as usize);
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == lo {
        return hi;
    }
    let tri = hi * (hi - 1) / 2;
    let base = if c0.suit() == c1.suit() { 13 } else { 91 };
    base + tri + lo
}

/// Hai lá đại diện cho mỗi lớp bài (theo đúng thứ tự class_index)
fn class_representatives() -> Vec<[Card; 2]> {
    let mut reps: Vec<Option<[Card; 2]>> = vec![None; PREFLOP_CLASSES];
    for hi in 0..13u8 {
        for lo in 0..=hi {
            let (rh, rl) = (Rank::from_u8(hi), Rank::from_u8(lo));
            if hi == lo {
                let h = [Card::new(rh, Suit::Spades), Card::new(rl, Suit::Hearts)];
                reps[class_index(h[0], h[1])] = Some(h);
            } else {
                let s = [Card::new(rh, Suit::Spades), Card::new(rl, Suit::Spades)];
                reps[class_index(s[0], s[1])] = Some(s);
                let o = [Card::new(rh, Suit::Spades), Card::new(rl, Suit::Hearts)];
                reps[class_index(o[0], o[1])] = Some(o);
            }
        }
    }
    reps.into_iter().map(|x| x.expect("class thiếu")).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflopEquityTable {
    pub samples: usize,
    /// [class * MAX_OPP + (num_opp - 1)]
    pub data: Vec<f32>,
}

impl PreflopEquityTable {
    /// Build song song bằng rayon (vài giây)
    pub fn build(samples: usize) -> Self {
        let reps = class_representatives();
        let rows: Vec<[f32; MAX_OPP]> = reps
            .par_iter()
            .enumerate()
            .map(|(cls, hole)| {
                let mut row = [0.0f32; MAX_OPP];
                for n in 1..=MAX_OPP {
                    let mut rng = XorShift::new(cls as u64 * 31 + n as u64 * 1_000_003 + 17);
                    row[n - 1] = monte_carlo_equity(*hole, &[], n, samples, &mut rng);
                }
                row
            })
            .collect();
        let data = rows.into_iter().flat_map(|r| r.into_iter()).collect();
        Self { samples, data }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> Option<Self> {
        let json = std::fs::read_to_string(path).ok()?;
        let t: Self = serde_json::from_str(&json).ok()?;
        if t.data.len() == PREFLOP_CLASSES * MAX_OPP {
            Some(t)
        } else {
            None
        }
    }

    #[inline]
    pub fn get(&self, c0: Card, c1: Card, num_opp: usize) -> f32 {
        let n = num_opp.clamp(1, MAX_OPP);
        self.data[class_index(c0, c1) * MAX_OPP + (n - 1)]
    }
}

static PREFLOP_TABLE: OnceLock<PreflopEquityTable> = OnceLock::new();

/// Lấy bảng preflop (load từ file; nếu chưa có thì build rồi lưu). Gọi sớm lúc khởi động
/// để lần build đầu không làm đứng GUI/training.
pub fn preflop_table() -> &'static PreflopEquityTable {
    PREFLOP_TABLE.get_or_init(|| {
        if let Some(t) = PreflopEquityTable::load(PREFLOP_TABLE_PATH) {
            return t;
        }
        let t = PreflopEquityTable::build(PREFLOP_BUILD_SAMPLES);
        let _ = t.save(PREFLOP_TABLE_PATH);
        t
    })
}

// ---------------------------------------------------------------- API chính

/// Equity so với `num_opp` đối thủ ngẫu nhiên. Preflop tra bảng, postflop Monte Carlo.
pub fn equity_vs_random(hole: [Card; 2], board: &[Card], num_opp: usize) -> f32 {
    if num_opp == 0 {
        return 1.0;
    }
    if board.is_empty() {
        return preflop_table().get(hole[0], hole[1], num_opp);
    }
    let mut rng = XorShift::new(seed_for(hole, board, num_opp));
    monte_carlo_equity(hole, board, num_opp, POSTFLOP_SAMPLES, &mut rng)
}

/// 5 đặc trưng: [flush draw, straight draw, board có đôi, max cùng chất trên board / 5, lá cao nhất board / 12]
pub fn board_and_draw_features(hole: [Card; 2], board: &[Card]) -> [f32; 5] {
    let mut all_mask = 0u16;
    let mut hole_mask = 0u16;
    let mut suit_cnt = [0u8; 4];
    let mut hole_suit = [0u8; 4];
    let mut board_suit = [0u8; 4];
    let mut board_rank_cnt = [0u8; 13];
    let mut board_high = 0u8;

    for c in hole {
        all_mask |= c.rank_bitmask();
        hole_mask |= c.rank_bitmask();
        suit_cnt[c.suit_val() as usize] += 1;
        hole_suit[c.suit_val() as usize] += 1;
    }
    for c in board {
        all_mask |= c.rank_bitmask();
        suit_cnt[c.suit_val() as usize] += 1;
        board_suit[c.suit_val() as usize] += 1;
        board_rank_cnt[c.rank_val() as usize] += 1;
        board_high = board_high.max(c.rank_val());
    }

    let drawing_street = board.len() == 3 || board.len() == 4;

    let flush_draw = drawing_street && (0..4).any(|s| suit_cnt[s] == 4 && hole_suit[s] >= 1);

    let mut made = false;
    let mut draw = false;
    let mut patterns = [0u16; 10];
    for w in 0..=8 {
        patterns[w] = 0b11111u16 << w;
    }
    patterns[9] = 0x100F; // wheel A-2-3-4-5
    for p in patterns {
        let n = (all_mask & p).count_ones();
        if n == 5 {
            made = true;
        } else if n == 4 && (hole_mask & p) != 0 {
            draw = true;
        }
    }
    let straight_draw = drawing_street && draw && !made;

    let paired = board_rank_cnt.iter().any(|&c| c >= 2);
    let max_suit = *board_suit.iter().max().unwrap_or(&0) as f32 / 5.0;
    let high = if board.is_empty() { 0.0 } else { board_high as f32 / 12.0 };

    [
        if flush_draw { 1.0 } else { 0.0 },
        if straight_draw { 1.0 } else { 0.0 },
        if paired { 1.0 } else { 0.0 },
        max_suit,
        high,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(s: &str) -> Card {
        Card::from_str_repr(s).unwrap()
    }

    #[test]
    fn test_class_index_unique_and_in_range() {
        let reps = class_representatives();
        assert_eq!(reps.len(), PREFLOP_CLASSES);
        let mut seen = vec![false; PREFLOP_CLASSES];
        for h in &reps {
            let i = class_index(h[0], h[1]);
            assert!(i < PREFLOP_CLASSES && !seen[i]);
            seen[i] = true;
        }
        // suited/offsuit khác lớp, thứ tự lá không quan trọng
        assert_ne!(class_index(c("As"), c("Ks")), class_index(c("As"), c("Kh")));
        assert_eq!(class_index(c("As"), c("Kh")), class_index(c("Kd"), c("Ac")));
    }

    #[test]
    fn test_known_equities() {
        let mut rng = XorShift::new(1);
        let aa = monte_carlo_equity([c("As"), c("Ah")], &[], 1, 20_000, &mut rng);
        assert!((aa - 0.85).abs() < 0.02, "AA vs 1 = {}", aa);
        let w = monte_carlo_equity([c("7s"), c("2h")], &[], 1, 20_000, &mut rng);
        assert!(w < 0.40, "72o vs 1 = {}", w);
        // nhiều đối thủ thì equity giảm
        let aa6 = monte_carlo_equity([c("As"), c("Ah")], &[], 6, 10_000, &mut rng);
        assert!(aa6 < aa);
    }

    #[test]
    fn test_postflop_deterministic_and_nuts() {
        let hole = [c("As"), c("Ah")];
        let board = [c("Ad"), c("Ac"), c("2h")];
        let e1 = equity_vs_random(hole, &board, 3);
        let e2 = equity_vs_random(hole, &board, 3);
        assert_eq!(e1, e2);
        assert!(e1 > 0.95);
    }

    #[test]
    fn test_draw_features() {
        // flush draw: 2 lá cơ trên tay + 2 lá cơ trên flop
        let f = board_and_draw_features([c("As"), c("Ks")], &[c("2s"), c("7s"), c("9h")]);
        assert_eq!(f[0], 0.0); // chỉ 4 lá spade? A,K,2,7 spade = 4 -> có draw
    }
}

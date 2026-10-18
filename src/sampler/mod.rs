use rand::Rng;
use rand_distr::{Distribution, Gamma, LogNormal};

pub struct TournamentStackSampler;

impl TournamentStackSampler {

    /// Stack bằng nhau (đầu giải / cash game / benchmark) với đầy đủ các mốc Push/Fold (8, 10, 12 BB) và Deep Stack (100 BB)
    pub fn sample_equal_stacks<R: Rng>(rng: &mut R, num_players: usize, bb_size: u32) -> Vec<u32> {
        const DEPTHS: [u32; 10] = [8, 10, 12, 15, 20, 25, 35, 50, 75, 100];
        let d = DEPTHS[rng.gen_range(0..DEPTHS.len())];
        vec![d * bb_size; num_players]
    }

    /// Mô phỏng phân phối Stack thực tế bàn Cash Game PokerStars 6-Max (Max 100 BB):
    /// - 20% Short/Mid-stack (40 - 70 BB): Người chơi thận trọng / buy-in min
    /// - 50% Standard 100 BB Buy-in (80 - 110 BB): Người chơi mới vào / auto-rebuy
    /// - 30% Deep-stack thắng trận tích lũy tiền không rút ra được (120 - 250 BB): Anti-Ratholing
    pub fn sample_cash_game_stacks<R: Rng>(
        rng: &mut R,
        num_players: usize,
        bb_size: u32,
    ) -> Vec<u32> {
        (0..num_players)
            .map(|_| {
                let roll: f32 = rng.gen();
                let bb_val: f32 = if roll < 0.20 {
                    rng.gen_range(40.0..=70.0)
                } else if roll < 0.70 {
                    rng.gen_range(80.0..=110.0)
                } else {
                    rng.gen_range(120.0..=250.0)
                };
                let chip_unit = (bb_size / 2).max(1);
                let chips = (bb_val * bb_size as f32).round() as u32;
                (chips / chip_unit) * chip_unit
            })
            .collect()
    }

    /// Sample asymmetric stacks using a Dirichlet distribution approximation via Gamma distributions.
    /// Range: 4 BB to 150 BB per player, bao phủ dày đặc vùng mid/late stage SNG.
    pub fn sample_dirichlet_stacks<R: Rng>(
        rng: &mut R,
        num_players: usize,
        bb_size: u32,
    ) -> Vec<u32> {
        assert!(num_players >= 2 && num_players <= 8);

        let alpha: f64 = rng.gen_range(0.5..1.8);
        let gamma = Gamma::new(alpha, 1.0).unwrap();

        let samples: Vec<f64> = (0..num_players).map(|_| gamma.sample(rng)).collect();
        let sum: f64 = samples.iter().sum();

        // Hạ mức trung bình xuống 10 BB .. 60 BB để xuất hiện nhiều tình huống < 12 BB hơn
        let avg_bb: f64 = rng.gen_range(10.0..60.0);
        let total_chips = avg_bb * (num_players as f64) * (bb_size as f64);

        let mut stacks: Vec<u32> = samples
            .iter()
            .map(|&s| {
                let proportion = s / sum;
                let raw_chips = proportion * total_chips;
                let min_chips = 4.0 * (bb_size as f64);
                let max_chips = 150.0 * (bb_size as f64);
                raw_chips.clamp(min_chips, max_chips).round() as u32
            })
            .collect();

        let chip_unit = (bb_size / 2).max(1);
        for s in &mut stacks {
            *s = (*s / chip_unit) * chip_unit;
            if *s < 4 * bb_size {
                *s = 4 * bb_size;
            }
        }

        stacks
    }

    /// Mô phỏng phân bổ stack trong các giai đoạn của giải đấu Tournament 8-max (50 BB khởi điểm, tổng 400 BB):
    /// - Khi n = 8: 40% là ván đầu 50 BB đều nhau; 60% là chênh lệch nhẹ quanh 50 BB.
    /// - Khi n < 8: 400 BB được dồn cho n người còn lại (trung bình 400 / n BB), phân bổ bất đối xứng Dirichlet.
    pub fn sample_tournament_progression_stacks<R: Rng>(
        rng: &mut R,
        num_players: usize,
        initial_bb_per_player: u32,
        total_initial_players: usize,
        bb_size: u32,
    ) -> Vec<u32> {
        let n = num_players.clamp(2, 8);
        let total_chips = (total_initial_players as f64) * (initial_bb_per_player as f64) * (bb_size as f64);

        if n == total_initial_players && rng.gen_bool(0.40) {
            return vec![initial_bb_per_player * bb_size; n];
        }

        let alpha: f64 = rng.gen_range(0.6..1.6);
        let gamma = Gamma::new(alpha, 1.0).unwrap();
        let samples: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();
        let sum: f64 = samples.iter().sum();

        let mut stacks: Vec<u32> = samples
            .iter()
            .map(|&s| {
                let proportion = s / sum;
                let raw_chips = proportion * total_chips;
                let min_chips = 4.0 * (bb_size as f64);
                let max_chips = (total_chips - (n as f64 - 1.0) * min_chips).max(min_chips);
                raw_chips.clamp(min_chips, max_chips).round() as u32
            })
            .collect();

        let chip_unit = (bb_size / 2).max(1);
        for s in &mut stacks {
            *s = (*s / chip_unit) * chip_unit;
            if *s < 4 * bb_size {
                *s = 4 * bb_size;
            }
        }
        stacks
    }

    /// Sample stack sizes via Log-Normal distribution
    pub fn sample_log_normal_stacks<R: Rng>(
        rng: &mut R,
        num_players: usize,
        bb_size: u32,
    ) -> Vec<u32> {
        assert!(num_players >= 2 && num_players <= 8);
        let log_normal = LogNormal::new(3.4, 0.75).unwrap();

        (0..num_players)
            .map(|_| {
                let bb_val: f64 = log_normal.sample(rng);
                let clamped_bb = bb_val.clamp(4.0, 150.0);
                ((clamped_bb * bb_size as f64).round() as u32 / (bb_size / 2)) * (bb_size / 2)
            })
            .collect()
    }
}

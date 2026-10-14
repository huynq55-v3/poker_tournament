use rand::Rng;
use rand_distr::{Distribution, Gamma, LogNormal};

pub struct TournamentStackSampler;

impl TournamentStackSampler {

    /// Stack bằng nhau (đầu giải / cash game / benchmark) với đầy đủ các mốc Push/Fold (8, 10, 12 BB)
    pub fn sample_equal_stacks<R: Rng>(rng: &mut R, num_players: usize, bb_size: u32) -> Vec<u32> {
        const DEPTHS: [u32; 9] = [8, 10, 12, 15, 20, 25, 35, 50, 75];
        let d = DEPTHS[rng.gen_range(0..DEPTHS.len())];
        vec![d * bb_size; num_players]
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

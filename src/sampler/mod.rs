use rand::Rng;
use rand_distr::{Distribution, Gamma};

pub struct TournamentStackSampler;

impl TournamentStackSampler {
    /// Stack bằng nhau với đầy đủ các mốc Push/Fold (8, 10, 12 BB) và Deep Stack (100 BB)
    pub fn sample_equal_stacks<R: Rng>(rng: &mut R, num_players: usize, bb_size: u32) -> Vec<u32> {
        const DEPTHS: [u32; 10] = [8, 10, 12, 15, 20, 25, 35, 50, 75, 100];
        let d = DEPTHS[rng.gen_range(0..DEPTHS.len())];
        vec![d * bb_size; num_players]
    }

    /// Phân phối Stack thực tế bàn Lounge / Real Coin Cash Game (2..=10 người):
    /// - 20% Short-stack: 12 - 39 BB
    /// - 45% Standard buy-in: 40 - 100 BB
    /// - 35% Deep-stack: 101 - 300 BB
    pub fn sample_lounge_stacks<R: Rng>(rng: &mut R, num_players: usize, bb_size: u32) -> Vec<u32> {
        assert!((2..=10).contains(&num_players));
        let chip_unit = (bb_size / 2).max(1);
        (0..num_players)
            .map(|_| {
                let roll: f32 = rng.gen();
                let bb_val: f32 = if roll < 0.20 {
                    rng.gen_range(12.0..=39.0)
                } else if roll < 0.65 {
                    rng.gen_range(40.0..=100.0)
                } else {
                    rng.gen_range(101.0..=300.0)
                };
                let chips = (bb_val * bb_size as f32).round() as u32;
                (chips / chip_unit) * chip_unit
            })
            .collect()
    }

    /// Sample asymmetric stacks using a Dirichlet distribution approximation via Gamma distributions (4..=150 BB).
    pub fn sample_dirichlet_stacks<R: Rng>(rng: &mut R, num_players: usize, bb_size: u32) -> Vec<u32> {
        assert!((2..=10).contains(&num_players));

        let alpha: f64 = rng.gen_range(0.5..1.8);
        let gamma = Gamma::new(alpha, 1.0).unwrap();

        let samples: Vec<f64> = (0..num_players).map(|_| gamma.sample(rng)).collect();
        let sum: f64 = samples.iter().sum();

        let avg_bb: f64 = rng.gen_range(10.0..60.0);
        let total_chips = avg_bb * (num_players as f64) * (bb_size as f64);

        let chip_unit = (bb_size / 2).max(1);
        samples
            .iter()
            .map(|&s| {
                let proportion = s / sum;
                let raw_chips = proportion * total_chips;
                let min_chips = 4.0 * (bb_size as f64);
                let max_chips = 150.0 * (bb_size as f64);
                let c = (raw_chips.clamp(min_chips, max_chips).round() as u32 / chip_unit) * chip_unit;
                c.max(4 * bb_size)
            })
            .collect()
    }

    /// Phân bổ stack theo tiến trình giải đấu Tournament (2..=10 người).
    pub fn sample_tournament_progression_stacks<R: Rng>(
        rng: &mut R,
        num_players: usize,
        initial_bb_per_player: u32,
        total_initial_players: usize,
        bb_size: u32,
    ) -> Vec<u32> {
        let n = num_players.clamp(2, 10);
        let total_chips = (total_initial_players as f64) * (initial_bb_per_player as f64) * (bb_size as f64);

        if n == total_initial_players && rng.gen_bool(0.40) {
            return vec![initial_bb_per_player * bb_size; n];
        }

        let alpha: f64 = rng.gen_range(0.6..1.6);
        let gamma = Gamma::new(alpha, 1.0).unwrap();
        let samples: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();
        let sum: f64 = samples.iter().sum();

        let chip_unit = (bb_size / 2).max(1);
        samples
            .iter()
            .map(|&s| {
                let proportion = s / sum;
                let raw_chips = proportion * total_chips;
                let min_chips = 4.0 * (bb_size as f64);
                let max_chips = (total_chips - (n as f64 - 1.0) * min_chips).max(min_chips);
                let c = (raw_chips.clamp(min_chips, max_chips).round() as u32 / chip_unit) * chip_unit;
                c.max(4 * bb_size)
            })
            .collect()
    }
}

use rand::Rng;
use rand_distr::{Distribution, Gamma, LogNormal};

pub struct TournamentStackSampler;

impl TournamentStackSampler {
    /// Sample asymmetric stacks using a Dirichlet distribution approximation via Gamma distributions.
    /// Range: 5 BB to 200 BB per player.
    pub fn sample_dirichlet_stacks<R: Rng>(
        rng: &mut R,
        num_players: usize,
        bb_size: u32,
    ) -> Vec<u32> {
        assert!(num_players >= 2 && num_players <= 8);

        // Alpha parameters: random concentration parameter alpha between 0.5 (high variance/chip disparity)
        // and 2.0 (more balanced stacks) to create realistic tournament dynamics
        let alpha: f64 = rng.gen_range(0.6..1.8);
        let gamma = Gamma::new(alpha, 1.0).unwrap();

        let samples: Vec<f64> = (0..num_players).map(|_| gamma.sample(rng)).collect();
        let sum: f64 = samples.iter().sum();

        // Target average stack in tournament: 20 BB to 80 BB
        let avg_bb: f64 = rng.gen_range(20.0..70.0);
        let total_chips = avg_bb * (num_players as f64) * (bb_size as f64);

        let mut stacks: Vec<u32> = samples
            .iter()
            .map(|&s| {
                let proportion = s / sum;
                let raw_chips = proportion * total_chips;
                // Clamp between 5 BB and 200 BB
                let min_chips = 5.0 * (bb_size as f64);
                let max_chips = 200.0 * (bb_size as f64);
                raw_chips.clamp(min_chips, max_chips).round() as u32
            })
            .collect();

        // Ensure stacks are multiples of small blind (e.g. bb_size / 2)
        let chip_unit = (bb_size / 2).max(1);
        for s in &mut stacks {
            *s = (*s / chip_unit) * chip_unit;
            if *s < 5 * bb_size {
                *s = 5 * bb_size;
            }
        }

        stacks
    }

    /// Sample stack sizes via Log-Normal distribution (representing deep-stack cash / late tournament MTT)
    pub fn sample_log_normal_stacks<R: Rng>(
        rng: &mut R,
        num_players: usize,
        bb_size: u32,
    ) -> Vec<u32> {
        assert!(num_players >= 2 && num_players <= 8);
        // LogNormal with mu = ln(40 BB), sigma = 0.75
        let log_normal = LogNormal::new(3.68, 0.75).unwrap();

        (0..num_players)
            .map(|_| {
                let bb_val: f64 = log_normal.sample(rng);
                let clamped_bb = bb_val.clamp(5.0, 200.0);
                ((clamped_bb * bb_size as f64).round() as u32 / (bb_size / 2)) * (bb_size / 2)
            })
            .collect()
    }
}

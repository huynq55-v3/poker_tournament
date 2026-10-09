use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::Street;
use crate::sampler::TournamentStackSampler;
use crossbeam_channel::{bounded, Receiver, Sender};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub player_idx: usize,
    pub features: Vec<f32>,
    pub action_mask: [bool; NUM_ACTIONS],
    pub action: usize,
    pub reward: f32,
}

pub trait Policy: Send + Sync {
    /// Select action given features and valid action mask
    fn select_action(&self, features: &[f32], mask: &ActionMask) -> Action;

    /// Get probability distribution over actions for GTO visualizer
    fn get_action_probs(&self, features: &[f32], mask: &ActionMask) -> [f32; NUM_ACTIONS];
}

/// Baseline Random Masked Policy
pub struct UniformRandomPolicy;

impl Policy for UniformRandomPolicy {
    fn select_action(&self, _features: &[f32], mask: &ActionMask) -> Action {
        let valid = mask.valid_actions();
        if valid.is_empty() {
            return Action::Fold;
        }
        let mut rng = thread_rng();
        *valid.choose(&mut rng).unwrap()
    }

    fn get_action_probs(&self, _features: &[f32], mask: &ActionMask) -> [f32; NUM_ACTIONS] {
        let mut probs = [0.0f32; NUM_ACTIONS];
        let valid = mask.valid_actions();
        if !valid.is_empty() {
            let p = 1.0 / (valid.len() as f32);
            for a in valid {
                probs[a.to_index()] = p;
            }
        }
        probs
    }
}

pub struct SelfPlayEngine {
    pub total_simulated_hands: Arc<AtomicU64>,
    pub running: Arc<AtomicBool>,
}

impl SelfPlayEngine {
    pub fn new() -> Self {
        Self {
            total_simulated_hands: Arc::new(AtomicU64::new(0)),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn run_parallel_simulation<P: Policy + 'static>(
        &self,
        policy: Arc<P>,
        num_workers: usize,
        hands_to_simulate: u64,
        buffer_capacity: usize,
    ) -> Receiver<Transition> {
        let (sender, receiver): (Sender<Transition>, Receiver<Transition>) = bounded(buffer_capacity);
        self.running.store(true, Ordering::SeqCst);

        let running = Arc::clone(&self.running);
        let hand_counter = Arc::clone(&self.total_simulated_hands);

        std::thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_workers)
                .build()
                .unwrap();

            pool.install(|| {
                (0..num_workers).into_par_iter().for_each(|_| {
                    let mut rng = thread_rng();
                    let bb_size = 100u32;
                    let sb_size = 50u32;

                    while running.load(Ordering::Relaxed) {
                        // ✅ FIX: chiếm slot nguyên tử, không bao giờ vượt hands_to_simulate
                        let claimed = hand_counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| {
                            if c < hands_to_simulate { Some(c + 1) } else { None }
                        });
                        if claimed.is_err() {
                            running.store(false, Ordering::Relaxed);
                            break;
                        }

                        let num_players = rand::Rng::gen_range(&mut rng, 2..=8);
                        let button_idx = rand::Rng::gen_range(&mut rng, 0..num_players);
                        let stacks = TournamentStackSampler::sample_dirichlet_stacks(&mut rng, num_players, bb_size);

                        let starting_stacks = stacks.clone();
                        let mut env = Environment::new(num_players, &stacks, bb_size, sb_size, button_idx);
                        env.reset_hand(&mut rng, button_idx);

                        let mut episode: Vec<(usize, Vec<f32>, ActionMask, Action)> = Vec::new();

                        while env.state.street != Street::Showdown {
                            let curr_p = env.state.current_player_idx;
                            let mask = env.get_action_mask();
                            if mask.0 == 0 {
                                break;
                            }
                            let features = env.state.encode_features(curr_p);

                            let action = policy.select_action(&features, &mask);
                            episode.push((curr_p, features, mask, action));

                            match env.step(action) {
                                Ok(true) => break,
                                Ok(false) => {}
                                Err(_) => break,
                            }
                        }

                        for (p_idx, feat, mask, act) in episode {
                            let start_s = starting_stacks[p_idx] as f32;
                            let end_s = env.state.players[p_idx].stack as f32;
                            let reward_bb = (end_s - start_s) / (bb_size as f32);

                            let transition = Transition {
                                player_idx: p_idx,
                                features: feat,
                                action_mask: mask.to_bool_array(),
                                action: act.to_index(),
                                reward: reward_bb,
                            };

                            if sender.send(transition).is_err() {
                                return;
                            }
                        }
                    }
                });
            });
        });

        receiver
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

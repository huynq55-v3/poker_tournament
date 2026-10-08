use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::neural_network::MLP;
use crate::self_play::Policy;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::thread_rng;
use std::sync::{Arc, RwLock};

pub struct DeepCFRPolicy {
    pub advantage_net: Arc<RwLock<MLP>>,
    pub is_greedy: bool,
}

impl DeepCFRPolicy {
    pub fn new(advantage_net: Arc<RwLock<MLP>>, is_greedy: bool) -> Self {
        Self {
            advantage_net,
            is_greedy,
        }
    }

    /// Compute Regret-Matching action probabilities from predicted advantages
    pub fn compute_regret_matching_probs(
        raw_advantages: &[f32],
        mask: &ActionMask,
    ) -> [f32; NUM_ACTIONS] {
        let mut probs = [0.0f32; NUM_ACTIONS];
        let mut pos_sum = 0.0f32;
        let mut legal_count = 0;

        for (a_idx, &adv) in raw_advantages.iter().enumerate() {
            if mask.is_index_valid(a_idx) {
                legal_count += 1;
                let pos_r = adv.max(0.0);
                probs[a_idx] = pos_r;
                pos_sum += pos_r;
            } else {
                probs[a_idx] = 0.0;
            }
        }

        if pos_sum > 1e-6 {
            for p in &mut probs {
                *p /= pos_sum;
            }
        } else if legal_count > 0 {
            // Fallback to uniform distribution over legal actions
            let uniform_p = 1.0 / (legal_count as f32);
            for a_idx in 0..NUM_ACTIONS {
                if mask.is_index_valid(a_idx) {
                    probs[a_idx] = uniform_p;
                }
            }
        }

        probs
    }
}

impl Policy for DeepCFRPolicy {
    fn select_action(&self, features: &[f32], mask: &ActionMask) -> Action {
        let probs = self.get_action_probs(features, mask);

        if self.is_greedy {
            // Pick argmax probability
            let mut best_idx = 0;
            let mut best_p = -1.0f32;
            for (i, &p) in probs.iter().enumerate() {
                if mask.is_index_valid(i) && p > best_p {
                    best_p = p;
                    best_idx = i;
                }
            }
            return Action::from_index(best_idx).unwrap_or(Action::Fold);
        }

        // Sample action according to regret-matching distribution
        let mut rng = thread_rng();
        if let Ok(dist) = WeightedIndex::new(&probs) {
            let chosen_idx = dist.sample(&mut rng);
            Action::from_index(chosen_idx).unwrap_or(Action::Fold)
        } else {
            // Fallback
            let valid = mask.valid_actions();
            if valid.is_empty() {
                Action::Fold
            } else {
                valid[0]
            }
        }
    }

    fn get_action_probs(&self, features: &[f32], mask: &ActionMask) -> [f32; NUM_ACTIONS] {
        let raw_advantages = {
            let net = self.advantage_net.read().unwrap();
            net.forward(features)
        };
        Self::compute_regret_matching_probs(&raw_advantages, mask)
    }
}

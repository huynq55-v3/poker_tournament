use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use rand::seq::SliceRandom;
use rand::thread_rng;

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

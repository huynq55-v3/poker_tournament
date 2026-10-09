use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::neural_network::MLP;
use crate::self_play::Policy;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::thread_rng;
use std::sync::{Arc, RwLock};

pub struct DeepCFRPolicy {
    pub advantage_net: Arc<RwLock<MLP>>,
    /// Mạng chiến lược trung bình (nếu có thì dùng để chơi; nếu None thì dùng regret matching)
    pub strategy_net: Arc<RwLock<Option<MLP>>>,
    pub is_greedy: bool,
}

impl DeepCFRPolicy {
    pub fn new(advantage_net: Arc<RwLock<MLP>>, is_greedy: bool) -> Self {
        Self {
            advantage_net,
            strategy_net: Arc::new(RwLock::new(None)),
            is_greedy,
        }
    }

    pub fn with_strategy(
        advantage_net: Arc<RwLock<MLP>>,
        strategy_net: Option<MLP>,
        is_greedy: bool,
    ) -> Self {
        Self {
            advantage_net,
            strategy_net: Arc::new(RwLock::new(strategy_net)),
            is_greedy,
        }
    }

    fn uniform_over_legal(mask: &ActionMask) -> [f32; NUM_ACTIONS] {
        let mut probs = [0.0f32; NUM_ACTIONS];
        let legal = mask.valid_actions();
        if !legal.is_empty() {
            let p = 1.0 / legal.len() as f32;
            for a in legal {
                probs[a.to_index()] = p;
            }
        }
        probs
    }

    /// Regret matching chuẩn: tỉ lệ theo regret dương; nếu không có regret dương
    /// -> phân phối ĐỀU trên các action hợp lệ (tổng luôn = 1).
    pub fn compute_regret_matching_probs(
        raw_advantages: &[f32],
        mask: &ActionMask,
    ) -> [f32; NUM_ACTIONS] {
        let mut probs = [0.0f32; NUM_ACTIONS];
        let mut pos_sum = 0.0f32;

        for (a_idx, &adv) in raw_advantages.iter().enumerate().take(NUM_ACTIONS) {
            if mask.is_index_valid(a_idx) {
                let pos_r = adv.max(0.0);
                probs[a_idx] = pos_r;
                pos_sum += pos_r;
            }
        }

        if pos_sum > 1e-6 {
            for p in &mut probs {
                *p /= pos_sum;
            }
            probs
        } else {
            Self::uniform_over_legal(mask)
        }
    }

    /// Output của strategy net -> xác suất hợp lệ (clamp >= 0, mask, chuẩn hóa)
    pub fn compute_strategy_probs(raw: &[f32], mask: &ActionMask) -> [f32; NUM_ACTIONS] {
        let mut probs = [0.0f32; NUM_ACTIONS];
        let mut sum = 0.0f32;
        for (a_idx, &v) in raw.iter().enumerate().take(NUM_ACTIONS) {
            if mask.is_index_valid(a_idx) {
                let p = v.max(0.0);
                probs[a_idx] = p;
                sum += p;
            }
        }
        if sum > 1e-6 {
            for p in &mut probs {
                *p /= sum;
            }
            probs
        } else {
            Self::uniform_over_legal(mask)
        }
    }
}

impl Policy for DeepCFRPolicy {
    fn select_action(&self, features: &[f32], mask: &ActionMask) -> Action {
        let probs = self.get_action_probs(features, mask);

        if self.is_greedy {
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

        let mut rng = thread_rng();
        if let Ok(dist) = WeightedIndex::new(&probs) {
            let chosen_idx = dist.sample(&mut rng);
            Action::from_index(chosen_idx).unwrap_or(Action::Fold)
        } else {
            let valid = mask.valid_actions();
            if valid.is_empty() {
                Action::Fold
            } else {
                valid[0]
            }
        }
    }

    fn get_action_probs(&self, features: &[f32], mask: &ActionMask) -> [f32; NUM_ACTIONS] {
        if let Ok(guard) = self.strategy_net.read() {
            if let Some(net) = guard.as_ref() {
                let raw = net.forward(features);
                return Self::compute_strategy_probs(&raw, mask);
            }
        }
        let raw_advantages = {
            let net = self.advantage_net.read().unwrap();
            net.forward(features)
        };
        Self::compute_regret_matching_probs(&raw_advantages, mask)
    }
}

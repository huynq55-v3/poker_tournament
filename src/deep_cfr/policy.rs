use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::neural_network::MLP;
use crate::self_play::Policy;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::thread_rng;
use std::sync::{Arc, RwLock};

pub struct DeepCFRPolicy {
    pub advantage_net: Arc<RwLock<MLP>>,
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

    /// Regret Matching chuẩn:
    /// - Nếu có Regret dương đáng kể (> 1e-4) -> chuẩn hóa tỉ lệ.
    /// - Nếu toàn bộ Regret âm hoặc sát 0 -> chọn hành động có Regret cao nhất (Argmax / EV cao nhất),
    ///   hoặc Sharp Softmax (T = 0.03) để tránh việc chia đều xác suất cho All-in/Fold rác.
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

        if pos_sum > 1e-4 {
            for p in &mut probs {
                *p /= pos_sum;
            }
            probs
        } else {
            // Khi toàn bộ Regret đều âm hoặc trong ngưỡng nhiễu:
            // Tìm hành động hợp lệ có giá trị regret lớn nhất (Argmax)
            let mut best_idx = None;
            let mut best_val = f32::NEG_INFINITY;

            for (a_idx, &adv) in raw_advantages.iter().enumerate().take(NUM_ACTIONS) {
                if mask.is_index_valid(a_idx) && adv > best_val {
                    best_val = adv;
                    best_idx = Some(a_idx);
                }
            }

            if let Some(idx) = best_idx {
                probs[idx] = 1.0;
            } else if let Some(first) = mask.valid_actions().first() {
                probs[first.to_index()] = 1.0;
            }

            probs
        }
    }

    /// Trích xuất xác suất từ Strategy Net (Average Strategy)
    pub fn compute_strategy_probs(raw: &[f32], mask: &ActionMask) -> [f32; NUM_ACTIONS] {
        let mut probs = [0.0f32; NUM_ACTIONS];
        let mut sum = 0.0f32;

        for (a_idx, &v) in raw.iter().enumerate().take(NUM_ACTIONS) {
            if mask.is_index_valid(a_idx) {
                // Chỉ lấy các giá trị dương thực sự, bỏ qua nhiễu cận 0
                let p = if v > 0.005 { v } else { 0.0 };
                probs[a_idx] = p;
                sum += p;
            }
        }

        if sum > 1e-4 {
            for p in &mut probs {
                *p /= sum;
            }
            probs
        } else {
            // Fallback: chọn hành động hợp lệ có giá trị lớn nhất trong output
            let mut best_idx = None;
            let mut best_val = f32::NEG_INFINITY;
            for (a_idx, &v) in raw.iter().enumerate().take(NUM_ACTIONS) {
                if mask.is_index_valid(a_idx) && v > best_val {
                    best_val = v;
                    best_idx = Some(a_idx);
                }
            }
            if let Some(idx) = best_idx {
                probs[idx] = 1.0;
            } else if let Some(first) = mask.valid_actions().first() {
                probs[first.to_index()] = 1.0;
            }
            probs
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
            mask.valid_actions().first().copied().unwrap_or(Action::Fold)
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

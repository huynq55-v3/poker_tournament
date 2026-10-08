use crate::deep_cfr::advantage_memory::{AdvantageMemory, AdvantageSample};
use crate::deep_cfr::policy::DeepCFRPolicy;
use crate::deep_cfr::strategy_memory::{StrategyMemory, StrategySample};
use crate::environment::action::{Action, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::neural_network::{LossType, MLP};
use crate::sampler::TournamentStackSampler;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::Rng;
use std::sync::{Arc, RwLock};

pub struct DeepCFRTrainer {
    pub advantage_net: Arc<RwLock<MLP>>,
    pub strategy_net: Arc<RwLock<MLP>>,
    pub adv_memory: AdvantageMemory,
    pub strat_memory: StrategyMemory,
    pub iteration: usize,
    pub lr: f32,
    pub weight_decay: f32,
}

impl DeepCFRTrainer {
    pub fn new(input_dim: usize, hidden_dims: &[usize], adv_capacity: usize, strat_capacity: usize) -> Self {
        let mut dims = Vec::with_capacity(hidden_dims.len() + 2);
        dims.push(input_dim);
        dims.extend_from_slice(hidden_dims);
        dims.push(NUM_ACTIONS);

        let adv_net = MLP::new(&dims);
        let strat_net = MLP::new(&dims);

        Self {
            advantage_net: Arc::new(RwLock::new(adv_net)),
            strategy_net: Arc::new(RwLock::new(strat_net)),
            adv_memory: AdvantageMemory::new(adv_capacity),
            strat_memory: StrategyMemory::new(strat_capacity),
            iteration: 0,
            lr: 0.001,
            weight_decay: 1e-4,
        }
    }

    /// External Sampling MCCFR Traversal
    pub fn traverse<R: Rng>(
        &mut self,
        env: &mut Environment,
        traversing_player: usize,
        starting_stack: f32,
        rng: &mut R,
        iteration_weight: f32,
    ) -> f32 {
        let curr_p = env.state.current_player_idx;
        let mask = env.get_action_mask();

        // Terminal condition or showdown
        if env.state.street == crate::environment::state::Street::Showdown {
            let final_stack = env.state.players[traversing_player].stack as f32;
            let payoff_bb = (final_stack - starting_stack) / (env.state.bb_size as f32);
            return payoff_bb;
        }

        // If player is folded or can't act
        let active_players = env.state.players.iter().filter(|p| p.is_in_hand()).count();
        if active_players <= 1 {
            let final_stack = env.state.players[traversing_player].stack as f32;
            let payoff_bb = (final_stack - starting_stack) / (env.state.bb_size as f32);
            return payoff_bb;
        }

        let features = env.state.encode_features(curr_p);

        // Predict current advantage estimates
        let raw_advs = {
            let net = self.advantage_net.read().unwrap();
            net.forward(&features)
        };
        let strat_probs = DeepCFRPolicy::compute_regret_matching_probs(&raw_advs, &mask);

        if curr_p == traversing_player {
            // Exploring node for the traversing player:
            // Evaluate each legal action to compute counterfactual regrets
            let valid_actions = mask.valid_actions();
            let mut action_values = [0.0f32; NUM_ACTIONS];
            let mut expected_node_val = 0.0f32;

            for &action in &valid_actions {
                let a_idx = action.to_index();
                let mut env_branch = env.clone();
                let _ended = env_branch.step(action).unwrap_or(true);

                let val = self.traverse(
                    &mut env_branch,
                    traversing_player,
                    starting_stack,
                    rng,
                    iteration_weight,
                );
                action_values[a_idx] = val;
                expected_node_val += strat_probs[a_idx] * val;
            }

            // Compute immediate counterfactual regrets: r(a) = v(a) - v
            let mut regrets = vec![0.0f32; NUM_ACTIONS];
            for &action in &valid_actions {
                let a_idx = action.to_index();
                regrets[a_idx] = action_values[a_idx] - expected_node_val;
            }

            // Store in Advantage Memory & Strategy Memory
            self.adv_memory.push(AdvantageSample {
                features: features.clone(),
                regrets,
                iteration_weight,
            });

            self.strat_memory.push(StrategySample {
                features,
                action_probs: strat_probs.to_vec(),
                iteration_weight,
            });

            expected_node_val
        } else {
            // Opponent's turn: External Sampling -> Sample a single action from strategy distribution
            let chosen_action = if let Ok(dist) = WeightedIndex::new(&strat_probs) {
                let a_idx = dist.sample(rng);
                Action::from_index(a_idx).unwrap_or(Action::Fold)
            } else {
                let valid = mask.valid_actions();
                if valid.is_empty() { Action::Fold } else { valid[0] }
            };

            let _ended = env.step(chosen_action).unwrap_or(true);
            self.traverse(
                env,
                traversing_player,
                starting_stack,
                rng,
                iteration_weight,
            )
        }
    }

    /// Run one Deep CFR training iteration
    pub fn step_iteration<R: Rng>(
        &mut self,
        num_traversals: usize,
        train_steps_per_iter: usize,
        batch_size: usize,
        rng: &mut R,
    ) -> (f32, f32) {
        self.iteration += 1;
        let iter_weight = self.iteration as f32;
        let bb_size = 100u32;
        let sb_size = 50u32;

        // 1. Data Collection via MCCFR Tree Traversals
        for _ in 0..num_traversals {
            // Domain Randomization: 2 to 6 players per traversal
            let num_players = rng.gen_range(2..=6);
            let button_idx = rng.gen_range(0..num_players);
            let stacks = TournamentStackSampler::sample_dirichlet_stacks(rng, num_players, bb_size);

            for traversing_p in 0..num_players {
                let mut env = Environment::new(num_players, &stacks, bb_size, sb_size, button_idx);
                env.reset_hand(rng, button_idx);
                let start_stack = env.state.players[traversing_p].stack as f32;

                self.traverse(&mut env, traversing_p, start_stack, rng, iter_weight);
            }
        }

        // 2. Train Advantage Network with Huber Loss
        let mut adv_loss = 0.0f32;
        if self.adv_memory.len() >= batch_size {
            for _ in 0..train_steps_per_iter {
                let (inputs, targets) = self.adv_memory.sample_batch(rng, batch_size);
                let mut net = self.advantage_net.write().unwrap();
                adv_loss += net.train_batch(&inputs, &targets, LossType::Huber, self.lr, self.weight_decay);
            }
            adv_loss /= train_steps_per_iter as f32;
        }

        // 3. Train Strategy Network with MSE/CrossEntropy Loss
        let mut strat_loss = 0.0f32;
        if self.strat_memory.len() >= batch_size {
            for _ in 0..train_steps_per_iter {
                let (inputs, targets) = self.strat_memory.sample_batch(rng, batch_size);
                let mut net = self.strategy_net.write().unwrap();
                strat_loss += net.train_batch(&inputs, &targets, LossType::MSE, self.lr, self.weight_decay);
            }
            strat_loss /= train_steps_per_iter as f32;
        }

        (adv_loss, strat_loss)
    }

    pub fn get_policy(&self, is_greedy: bool) -> DeepCFRPolicy {
        DeepCFRPolicy::new(Arc::clone(&self.advantage_net), is_greedy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deep_cfr_step_iteration() {
        let mut rng = rand::thread_rng();
        let mut trainer = DeepCFRTrainer::new(126, &[32, 32], 1000, 1000);
        let (adv_loss, strat_loss) = trainer.step_iteration(5, 2, 8, &mut rng);
        assert!(trainer.adv_memory.len() > 0);
        assert!(trainer.strat_memory.len() > 0);
        println!("Adv Loss: {}, Strat Loss: {}", adv_loss, strat_loss);
    }
}


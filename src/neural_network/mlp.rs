use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossType {
    MSE,
    Huber,
    CrossEntropy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<f32>, // row-major: [out_features * in_features]
    pub bias: Vec<f32>,    // [out_features]

    #[serde(skip)]
    pub grad_weights: Vec<f32>,
    #[serde(skip)]
    pub grad_bias: Vec<f32>,

    // Adam optimizer parameters
    #[serde(skip)]
    pub m_w: Vec<f32>,
    #[serde(skip)]
    pub v_w: Vec<f32>,
    #[serde(skip)]
    pub m_b: Vec<f32>,
    #[serde(skip)]
    pub v_b: Vec<f32>,
}

impl DenseLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        // He (Kaiming) initialization: std = sqrt(2 / in_features)
        let std_dev = (2.0f32 / in_features as f32).sqrt();

        let total_weights = in_features * out_features;
        let mut weights = Vec::with_capacity(total_weights);
        for _ in 0..total_weights {
            let u1: f32 = rng.gen_range(0.0001..1.0);
            let u2: f32 = rng.gen_range(0.0001..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            weights.push(z0 * std_dev);
        }

        let bias = vec![0.0f32; out_features];

        DenseLayer {
            in_features,
            out_features,
            weights,
            bias,
            grad_weights: vec![0.0; total_weights],
            grad_bias: vec![0.0; out_features],
            m_w: vec![0.0; total_weights],
            v_w: vec![0.0; total_weights],
            m_b: vec![0.0; out_features],
            v_b: vec![0.0; out_features],
        }
    }

    #[inline(always)]
    pub fn forward_single(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_features);
        debug_assert_eq!(output.len(), self.out_features);

        for o in 0..self.out_features {
            let w_offset = o * self.in_features;
            let mut sum = self.bias[o];
            for i in 0..self.in_features {
                sum += self.weights[w_offset + i] * input[i];
            }
            output[o] = sum;
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);
    }

    pub fn adam_update(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: usize,
        weight_decay: f32,
    ) {
        let t_f = t as f32;
        let b1_t = 1.0 - beta1.powf(t_f);
        let b2_t = 1.0 - beta2.powf(t_f);

        // Update weights
        for i in 0..self.weights.len() {
            let g = self.grad_weights[i] + weight_decay * self.weights[i];
            self.m_w[i] = beta1 * self.m_w[i] + (1.0 - beta1) * g;
            self.v_w[i] = beta2 * self.v_w[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m_w[i] / b1_t;
            let v_hat = self.v_w[i] / b2_t;

            self.weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        // Update bias
        for o in 0..self.bias.len() {
            let g = self.grad_bias[o];
            self.m_b[o] = beta1 * self.m_b[o] + (1.0 - beta1) * g;
            self.v_b[o] = beta2 * self.v_b[o] + (1.0 - beta2) * g * g;

            let m_hat = self.m_b[o] / b1_t;
            let v_hat = self.v_b[o] / b2_t;

            self.bias[o] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

/// Multi-Layer Perceptron (MLP) with ReLU activations and Linear/Softmax outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLP {
    pub layers: Vec<DenseLayer>,
    pub step_count: usize,
}

impl MLP {
    pub fn new(layer_dims: &[usize]) -> Self {
        assert!(layer_dims.len() >= 2);
        let mut layers = Vec::new();
        for i in 0..layer_dims.len() - 1 {
            layers.push(DenseLayer::new(layer_dims[i], layer_dims[i + 1]));
        }
        MLP {
            layers,
            step_count: 0,
        }
    }

    /// High-speed inference for a single sample (tree traversal)
    #[inline(always)]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        let num_layers = self.layers.len();

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let mut next = vec![0.0f32; layer.out_features];
            layer.forward_single(&current, &mut next);

            // Hidden layers use LeakyReLU activation (slope 0.01)
            if l_idx < num_layers - 1 {
                for val in &mut next {
                    if *val < 0.0 {
                        *val *= 0.01;
                    }
                }
            }
            current = next;
        }

        current
    }

    /// Forward pass storing activations for backpropagation
    fn forward_pass_with_cache(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        let mut post_activations = Vec::with_capacity(self.layers.len() + 1);

        post_activations.push(input.to_vec());
        let num_layers = self.layers.len();

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let curr_in = post_activations.last().unwrap();
            let mut pre = vec![0.0f32; layer.out_features];
            layer.forward_single(curr_in, &mut pre);

            let mut post = pre.clone();
            if l_idx < num_layers - 1 {
                for val in &mut post {
                    if *val < 0.0 {
                        *val *= 0.01;
                    }
                }
            }
            pre_activations.push(pre);
            post_activations.push(post);
        }

        (pre_activations, post_activations)
    }

    /// Train a mini-batch using Adam optimizer and return the average loss
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        loss_type: LossType,
        lr: f32,
        weight_decay: f32,
    ) -> f32 {
        let batch_size = inputs.len();
        if batch_size == 0 {
            return 0.0;
        }

        for layer in &mut self.layers {
            layer.zero_grad();
        }

        let mut total_loss = 0.0f32;
        let num_layers = self.layers.len();

        for b in 0..batch_size {
            let (pre_acts, post_acts) = self.forward_pass_with_cache(&inputs[b]);
            let output = post_acts.last().unwrap();
            let target = &targets[b];

            // 1. Compute loss & gradient at output layer
            let mut delta = vec![0.0f32; output.len()];
            for o in 0..output.len() {
                let diff = output[o] - target[o];
                match loss_type {
                    LossType::MSE => {
                        total_loss += 0.5 * diff * diff;
                        delta[o] = diff;
                    }
                    LossType::Huber => {
                        let delta_huber = 1.0f32;
                        if diff.abs() <= delta_huber {
                            total_loss += 0.5 * diff * diff;
                            delta[o] = diff;
                        } else {
                            total_loss += delta_huber * (diff.abs() - 0.5 * delta_huber);
                            delta[o] = delta_huber * diff.signum();
                        }
                    }
                    LossType::CrossEntropy => {
                        total_loss -= target[o] * (output[o].max(1e-7)).ln();
                        delta[o] = output[o] - target[o];
                    }
                }
            }

            // 2. Backpropagation through layers
            for l in (0..num_layers).rev() {
                let layer_in = &post_acts[l];
                let layer = &mut self.layers[l];

                // Accumulate gradients for weights & bias
                for o in 0..layer.out_features {
                    let d = delta[o];
                    layer.grad_bias[o] += d;
                    let w_offset = o * layer.in_features;
                    for i in 0..layer.in_features {
                        layer.grad_weights[w_offset + i] += d * layer_in[i];
                    }
                }

                // If not the first layer, propagate delta to previous layer
                if l > 0 {
                    let prev_layer_out_size = layer.in_features;
                    let mut prev_delta = vec![0.0f32; prev_layer_out_size];

                    for i in 0..prev_layer_out_size {
                        let mut sum = 0.0f32;
                        for o in 0..layer.out_features {
                            sum += layer.weights[o * layer.in_features + i] * delta[o];
                        }
                        // Derivative of LeakyReLU(0.01)
                        let pre_val = pre_acts[l - 1][i];
                        let act_grad = if pre_val > 0.0 { 1.0 } else { 0.01 };
                        prev_delta[i] = sum * act_grad;
                    }
                    delta = prev_delta;
                }
            }
        }

        // Normalize gradients by batch size
        let scale = 1.0 / (batch_size as f32);
        for layer in &mut self.layers {
            for g in &mut layer.grad_weights {
                *g *= scale;
            }
            for g in &mut layer.grad_bias {
                *g *= scale;
            }
        }

        // Step Adam optimizer
        self.step_count += 1;
        let t = self.step_count;
        for layer in &mut self.layers {
            layer.adam_update(lr, 0.9, 0.999, 1e-8, t, weight_decay);
        }

        total_loss / (batch_size as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_training_convergence() {
        let mut mlp = MLP::new(&[4, 16, 2]);
        let inputs = vec![
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
        ];
        let targets = vec![
            vec![1.0, -1.0],
            vec![-1.0, 1.0],
            vec![0.5, -0.5],
            vec![-0.5, 0.5],
        ];

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..100 {
            let loss = mlp.train_batch(&inputs, &targets, LossType::MSE, 0.05, 0.0);
            if step == 0 {
                initial_loss = loss;
            }
            final_loss = loss;
        }

        assert!(final_loss < initial_loss, "Final loss ({}) should be lower than initial loss ({})", final_loss, initial_loss);
    }
}


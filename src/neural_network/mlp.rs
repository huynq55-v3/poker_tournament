use rand::Rng;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

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
        // He (Kaiming) initialization
        let std_dev = (2.0f32 / in_features as f32).sqrt();

        let total_weights = in_features * out_features;
        let mut weights = Vec::with_capacity(total_weights);
        for _ in 0..total_weights {
            let u1: f32 = rng.gen_range(0.0001..1.0);
            let u2: f32 = rng.gen_range(0.0001..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            weights.push(z0 * std_dev);
        }

        DenseLayer {
            in_features,
            out_features,
            weights,
            bias: vec![0.0f32; out_features],
            grad_weights: vec![0.0; total_weights],
            grad_bias: vec![0.0; out_features],
            m_w: vec![0.0; total_weights],
            v_w: vec![0.0; total_weights],
            m_b: vec![0.0; out_features],
            v_b: vec![0.0; out_features],
        }
    }

    /// ✅ FIX: sau khi load từ file, các buffer #[serde(skip)] rỗng -> cấp phát lại để train tiếp được
    pub fn ensure_buffers(&mut self) {
        let nw = self.in_features * self.out_features;
        let nb = self.out_features;
        if self.grad_weights.len() != nw {
            self.grad_weights = vec![0.0; nw];
        }
        if self.m_w.len() != nw {
            self.m_w = vec![0.0; nw];
        }
        if self.v_w.len() != nw {
            self.v_w = vec![0.0; nw];
        }
        if self.grad_bias.len() != nb {
            self.grad_bias = vec![0.0; nb];
        }
        if self.m_b.len() != nb {
            self.m_b = vec![0.0; nb];
        }
        if self.v_b.len() != nb {
            self.v_b = vec![0.0; nb];
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

        for i in 0..self.weights.len() {
            let g = self.grad_weights[i] + weight_decay * self.weights[i];
            self.m_w[i] = beta1 * self.m_w[i] + (1.0 - beta1) * g;
            self.v_w[i] = beta2 * self.v_w[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m_w[i] / b1_t;
            let v_hat = self.v_w[i] / b2_t;
            self.weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

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

/// Multi-Layer Perceptron (MLP) with LeakyReLU hidden activations and Linear outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLP {
    pub layers: Vec<DenseLayer>,
    pub step_count: usize,
}

impl MLP {

        /// [in, out1, out2, ...] để kiểm tra tương thích khi nạp model
    pub fn layer_dims(&self) -> Vec<usize> {
        let mut d = Vec::with_capacity(self.layers.len() + 1);
        if let Some(first) = self.layers.first() {
            d.push(first.in_features);
        }
        for l in &self.layers {
            d.push(l.out_features);
        }
        d
    }

    /// Xóa trạng thái Adam và bộ đếm bước (để bias correction đúng khi train tiếp)
    pub fn reset_optimizer(&mut self) {
        self.step_count = 0;
        for l in &mut self.layers {
            l.ensure_buffers();
            l.grad_weights.fill(0.0);
            l.grad_bias.fill(0.0);
            l.m_w.fill(0.0);
            l.v_w.fill(0.0);
            l.m_b.fill(0.0);
            l.v_b.fill(0.0);
        }
    }
    
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

    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let mut mlp: MLP = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        for layer in &mut mlp.layers {
            layer.ensure_buffers();
        }
        Ok(mlp)
    }

    /// Input dimension của mạng (để kiểm tra tương thích model)
    pub fn input_dim(&self) -> usize {
        self.layers.first().map(|l| l.in_features).unwrap_or(0)
    }

    /// High-speed inference for a single sample
    #[inline(always)]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        let num_layers = self.layers.len();

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let mut next = vec![0.0f32; layer.out_features];
            layer.forward_single(&current, &mut next);

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

        /// Forward + backward cho 1 mẫu, cộng gradient vào `g`, trả về loss của mẫu.
    fn sample_backward(
        &self,
        input: &[f32],
        target: &[f32],
        w: f32,
        mask: Option<&[bool]>,
        loss_type: LossType,
        g: &mut Grads,
    ) -> f32 {
        let (pre_acts, post_acts) = self.forward_pass_with_cache(input);
        let output = post_acts.last().unwrap();
        let num_layers = self.layers.len();

        let mut total = 0.0f32;
        let mut delta = vec![0.0f32; output.len()];
        for o in 0..output.len() {
            if let Some(m) = mask {
                if !m[o] {
                    continue;
                }
            }
            let diff = output[o] - target[o];
            match loss_type {
                LossType::MSE => {
                    total += w * 0.5 * diff * diff;
                    delta[o] = w * diff;
                }
                LossType::Huber => {
                    let d = 1.0f32;
                    if diff.abs() <= d {
                        total += w * 0.5 * diff * diff;
                        delta[o] = w * diff;
                    } else {
                        total += w * d * (diff.abs() - 0.5 * d);
                        delta[o] = w * d * diff.signum();
                    }
                }
                LossType::CrossEntropy => {
                    total -= w * target[o] * (output[o].max(1e-7)).ln();
                    delta[o] = w * (output[o] - target[o]);
                }
            }
        }

        for l in (0..num_layers).rev() {
            let layer_in = &post_acts[l];
            let layer = &self.layers[l];

            for o in 0..layer.out_features {
                let d = delta[o];
                g.b[l][o] += d;
                let off = o * layer.in_features;
                for i in 0..layer.in_features {
                    g.w[l][off + i] += d * layer_in[i];
                }
            }

            if l > 0 {
                let mut prev_delta = vec![0.0f32; layer.in_features];
                for i in 0..layer.in_features {
                    let mut sum = 0.0f32;
                    for o in 0..layer.out_features {
                        sum += layer.weights[o * layer.in_features + i] * delta[o];
                    }
                    let act_grad = if pre_acts[l - 1][i] > 0.0 { 1.0 } else { 0.01 };
                    prev_delta[i] = sum * act_grad;
                }
                delta = prev_delta;
            }
        }
        total
    }

    /// Train 1 mini-batch, gradient tính SONG SONG theo mẫu (rayon).
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        sample_weights: Option<&[f32]>,
        output_masks: Option<&[Vec<bool>]>,
        loss_type: LossType,
        lr: f32,
        weight_decay: f32,
    ) -> f32 {
        let n = inputs.len();
        if n == 0 {
            return 0.0;
        }
        for layer in &mut self.layers {
            layer.ensure_buffers();
        }

        let (grads, loss) = {
            let this: &MLP = &*self;
            (0..n)
                .into_par_iter()
                .fold(
                    || (Grads::zeros(&this.layers), 0.0f32),
                    |mut acc, b| {
                        let w = sample_weights.map(|ws| ws[b]).unwrap_or(1.0);
                        let m = output_masks.map(|ms| ms[b].as_slice());
                        acc.1 += this.sample_backward(&inputs[b], &targets[b], w, m, loss_type, &mut acc.0);
                        acc
                    },
                )
                .reduce(
                    || (Grads::zeros(&this.layers), 0.0f32),
                    |mut a, b| {
                        a.0.add(&b.0);
                        a.1 += b.1;
                        a
                    },
                )
        };

        let scale = 1.0 / n as f32;
        for (l, layer) in self.layers.iter_mut().enumerate() {
            for (dst, src) in layer.grad_weights.iter_mut().zip(&grads.w[l]) {
                *dst = *src * scale;
            }
            for (dst, src) in layer.grad_bias.iter_mut().zip(&grads.b[l]) {
                *dst = *src * scale;
            }
        }

        self.step_count += 1;
        let t = self.step_count;
        for layer in &mut self.layers {
            layer.adam_update(lr, 0.9, 0.999, 1e-8, t, weight_decay);
        }

        loss * scale
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
            let loss = mlp.train_batch(&inputs, &targets, None, None, LossType::MSE, 0.05, 0.0);
            if step == 0 {
                initial_loss = loss;
            }
            final_loss = loss;
        }

        assert!(final_loss < initial_loss, "Final loss ({}) should be lower than initial loss ({})", final_loss, initial_loss);
    }

    #[test]
    fn test_train_after_reload_does_not_panic() {
        let mlp = MLP::new(&[3, 8, 2]);
        let json = serde_json::to_string(&mlp).unwrap();
        let mut loaded: MLP = serde_json::from_str(&json).unwrap();
        let inputs = vec![vec![0.1, 0.2, 0.3]];
        let targets = vec![vec![1.0, 0.0]];
        let _ = loaded.train_batch(&inputs, &targets, None, None, LossType::MSE, 0.01, 0.0);
    }
}

/// Gradient của cả mạng, dùng làm bộ cộng dồn cho từng luồng
struct Grads {
    w: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
}

impl Grads {
    fn zeros(layers: &[DenseLayer]) -> Self {
        Grads {
            w: layers.iter().map(|l| vec![0.0; l.in_features * l.out_features]).collect(),
            b: layers.iter().map(|l| vec![0.0; l.out_features]).collect(),
        }
    }
    fn add(&mut self, o: &Grads) {
        for (a, b) in self.w.iter_mut().zip(&o.w) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += *y;
            }
        }
        for (a, b) in self.b.iter_mut().zip(&o.b) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += *y;
            }
        }
    }
}

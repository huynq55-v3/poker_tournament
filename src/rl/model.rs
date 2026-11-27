//! Bước 2 của bot RL: transformer với hai đầu ra (policy + value).
//!
//! ```text
//!  token ─┬─ Embedding(kind) ┐
//!         ├─ Embedding(rank) │
//!         ├─ Embedding(suit) │
//!         ├─ Embedding(actor)├─ cộng ─ LayerNorm ─ TransformerEncoder (pre-norm, che PAD)
//!         ├─ Embedding(street)                         │
//!         ├─ Embedding(action)                         └─ lấy đầu ra tại token STATE (vị trí 0)
//!         ├─ Embedding(vị trí)                              ├─ đầu policy → 6 logit (che hành động không hợp lệ)
//!         └─ Linear(đặc trưng liên tục)                     └─ đầu value  → 1 số (EV, đơn vị 100 BB)
//! ```
//!
//! Đầu value dự đoán EV từ góc nhìn người đang hành động, đơn vị 100 BB (stack đầy = 1.0),
//! nên nằm trong khoảng [−1, 1] với heads-up 100 BB.

use crate::environment::{ActionMask, NUM_ACTIONS};
use crate::rl::encoding::{
    EncodedState, ACTION_VOCAB, ACTOR_VOCAB, CONT_DIM, DISCRETE_FIELDS, KIND_VOCAB, MAX_TOKENS, RANK_VOCAB,
    STREET_VOCAB, SUIT_VOCAB,
};
use burn::config::Config;
use burn::module::Module;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, log_softmax};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

/// Giá trị gán cho logit của hành động không hợp lệ (exp ≈ 0, vẫn hữu hạn để không sinh NaN).
const MASKED_LOGIT: f32 = -1.0e9;

#[derive(Config, Debug)]
pub struct PokerTransformerConfig {
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 256)]
    pub d_ff: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 4)]
    pub n_layers: usize,
}

#[derive(Module, Debug)]
pub struct PokerTransformer {
    emb_kind: Embedding,
    emb_rank: Embedding,
    emb_suit: Embedding,
    emb_actor: Embedding,
    emb_street: Embedding,
    emb_action: Embedding,
    emb_pos: Embedding,
    cont_proj: Linear,
    input_norm: LayerNorm,
    encoder: TransformerEncoder,
    out_norm: LayerNorm,
    policy_hidden: Linear,
    policy_out: Linear,
    value_hidden: Linear,
    value_out: Linear,
}

/// Một batch trạng thái đã chuyển thành tensor.
///
/// Độ dài chuỗi L = độ dài trạng thái dài nhất trong batch (không độn tới `MAX_TOKENS`):
/// phần lớn trạng thái chỉ 8–15 token nên cách này nhanh hơn vài lần.
#[derive(Debug, Clone)]
pub struct StateBatch {
    /// 6 trường rời rạc, mỗi trường [B, L].
    pub fields: [Tensor<2, Int>; DISCRETE_FIELDS],
    /// Đặc trưng liên tục [B, L, CONT_DIM].
    pub cont: Tensor<3>,
    /// true = PAD [B, L].
    pub pad: Tensor<2, Bool>,
    /// true = hành động KHÔNG hợp lệ [B, NUM_ACTIONS].
    pub invalid: Tensor<2, Bool>,
}

impl StateBatch {
    pub fn new(states: &[&EncodedState], masks: &[ActionMask], device: &Device) -> Self {
        assert_eq!(states.len(), masks.len());
        assert!(!states.is_empty(), "batch rỗng");
        let b = states.len();
        let l = states.iter().map(|s| s.len).max().unwrap_or(1).max(1);

        let mut disc = vec![0i64; b * l * DISCRETE_FIELDS];
        let mut cont = vec![0f32; b * l * CONT_DIM];
        let mut pad = vec![false; b * l];
        let mut invalid = vec![false; b * NUM_ACTIONS];
        let mut disc_full = [0i64; MAX_TOKENS * DISCRETE_FIELDS];
        let mut cont_full = [0f32; MAX_TOKENS * CONT_DIM];
        let mut pad_full = [false; MAX_TOKENS];
        for (i, (s, m)) in states.iter().zip(masks).enumerate() {
            s.write_discrete(&mut disc_full);
            s.write_cont(&mut cont_full);
            s.write_pad_mask(&mut pad_full);
            disc[i * l * DISCRETE_FIELDS..(i + 1) * l * DISCRETE_FIELDS]
                .copy_from_slice(&disc_full[..l * DISCRETE_FIELDS]);
            cont[i * l * CONT_DIM..(i + 1) * l * CONT_DIM].copy_from_slice(&cont_full[..l * CONT_DIM]);
            pad[i * l..(i + 1) * l].copy_from_slice(&pad_full[..l]);
            assert!(m.0 != 0, "trạng thái không có hành động hợp lệ nào");
            for a in 0..NUM_ACTIONS {
                invalid[i * NUM_ACTIONS + a] = !m.is_index_valid(a);
            }
        }

        // Tách [B, L, F] thành F tensor [B, L].
        let fields: [Tensor<2, Int>; DISCRETE_FIELDS] = std::array::from_fn(|f| {
            let col: Vec<i64> = disc.iter().skip(f).step_by(DISCRETE_FIELDS).copied().collect();
            Tensor::<2, Int>::from_data(TensorData::new(col, [b, l]), device)
        });

        StateBatch {
            fields,
            cont: Tensor::<3>::from_data(TensorData::new(cont, [b, l, CONT_DIM]), device),
            pad: Tensor::<2, Bool>::from_data(TensorData::new(pad, [b, l]), device),
            invalid: Tensor::<2, Bool>::from_data(TensorData::new(invalid, [b, NUM_ACTIONS]), device),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.cont.dims()[0]
    }
}

/// Đầu ra của mô hình.
#[derive(Debug, Clone)]
pub struct PolicyValue {
    /// log π(a|s) [B, NUM_ACTIONS]; hành động không hợp lệ ≈ −1e9.
    pub log_probs: Tensor<2>,
    /// EV ước lượng [B], đơn vị 100 BB.
    pub value: Tensor<1>,
}

impl PokerTransformerConfig {
    pub fn init(&self, device: &Device) -> PokerTransformer {
        let d = self.d_model;
        let emb = |n: usize| EmbeddingConfig::new(n, d).init(device);
        PokerTransformer {
            emb_kind: emb(KIND_VOCAB),
            emb_rank: emb(RANK_VOCAB),
            emb_suit: emb(SUIT_VOCAB),
            emb_actor: emb(ACTOR_VOCAB),
            emb_street: emb(STREET_VOCAB),
            emb_action: emb(ACTION_VOCAB),
            emb_pos: emb(MAX_TOKENS),
            cont_proj: LinearConfig::new(CONT_DIM, d).init(device),
            input_norm: LayerNormConfig::new(d).init(device),
            encoder: TransformerEncoderConfig::new(d, self.d_ff, self.n_heads, self.n_layers)
                .with_dropout(0.0)
                .with_norm_first(true)
                .init(device),
            out_norm: LayerNormConfig::new(d).init(device),
            policy_hidden: LinearConfig::new(d, d).init(device),
            policy_out: LinearConfig::new(d, NUM_ACTIONS).init(device),
            value_hidden: LinearConfig::new(d, d).init(device),
            value_out: LinearConfig::new(d, 1).init(device),
        }
    }
}

impl PokerTransformer {
    pub fn forward(&self, batch: &StateBatch) -> PolicyValue {
        let [b, l] = batch.pad.dims();
        let device = batch.cont.device();
        let [f_kind, f_rank, f_suit, f_actor, f_street, f_action] = batch.fields.clone();

        let pos = Tensor::<1, Int>::arange(0..l as i64, &device).reshape([1, l]).expand([b, l]);
        let x = self.emb_kind.forward(f_kind)
            + self.emb_rank.forward(f_rank)
            + self.emb_suit.forward(f_suit)
            + self.emb_actor.forward(f_actor)
            + self.emb_street.forward(f_street)
            + self.emb_action.forward(f_action)
            + self.emb_pos.forward(pos)
            + self.cont_proj.forward(batch.cont.clone());
        let x = self.input_norm.forward(x);

        let h = self.encoder.forward(TransformerEncoderInput::new(x).mask_pad(batch.pad.clone()));
        let d = h.dims()[2];
        // Token STATE luôn ở vị trí 0 và không bao giờ là PAD.
        let pooled = self.out_norm.forward(h.slice([0..b, 0..1, 0..d]).reshape([b, d]));

        let logits = self.policy_out.forward(gelu(self.policy_hidden.forward(pooled.clone())));
        let logits = logits.mask_fill(batch.invalid.clone(), MASKED_LOGIT);
        let log_probs = log_softmax(logits, 1);

        let value = self.value_out.forward(gelu(self.value_hidden.forward(pooled))).reshape([b]);
        PolicyValue { log_probs, value }
    }
}

/// Kết quả suy luận cho một trạng thái, đã đưa về CPU.
#[derive(Debug, Clone, Copy)]
pub struct Inference {
    pub probs: [f32; NUM_ACTIONS],
    pub value: f32,
}

impl PokerTransformer {
    /// Suy luận một batch, trả kết quả dạng mảng thường (dùng khi tự chơi / đấu arena).
    pub fn infer(&self, states: &[&EncodedState], masks: &[ActionMask], device: &Device) -> Vec<Inference> {
        let out = self.forward(&StateBatch::new(states, masks, device));
        let probs = out.log_probs.exp().into_data().try_to_vec::<f32>().expect("probs f32");
        let values = out.value.into_data().try_to_vec::<f32>().expect("value f32");
        probs
            .chunks_exact(NUM_ACTIONS)
            .zip(values)
            .map(|(p, v)| {
                let mut arr = [0f32; NUM_ACTIONS];
                arr.copy_from_slice(p);
                Inference { probs: arr, value: v }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{Action, Environment};
    use crate::rl::encoding::encode;
    use burn::optim::{AdamConfig, GradientsParams};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn small_cfg() -> PokerTransformerConfig {
        PokerTransformerConfig::new().with_d_model(32).with_d_ff(64).with_n_heads(2).with_n_layers(2)
    }

    /// Thu thập các trạng thái quyết định từ vài ván ngẫu nhiên (độ dài chuỗi khác nhau).
    fn sample_states(n: usize, seed: u64) -> Vec<(EncodedState, ActionMask)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::new();
        let mut hand = 0u64;
        while out.len() < n {
            let mut env = Environment::new(2, &[10_000, 10_000], 100, 50, 0);
            env.reset_hand(&mut StdRng::seed_from_u64(seed * 1000 + hand), (hand % 2) as usize);
            hand += 1;
            loop {
                let p = env.state.current_player_idx;
                let mask = env.get_action_mask();
                out.push((encode(&env.state, p), mask));
                let valid = mask.valid_actions();
                if env.step(valid[rng.gen_range(0..valid.len())]).unwrap() || out.len() >= n {
                    break;
                }
            }
        }
        out
    }

    #[test]
    fn forward_shapes_probabilities_and_masking() {
        let dev = Device::flex();
        let model = small_cfg().init(&dev);
        let data = sample_states(16, 1);
        let states: Vec<&EncodedState> = data.iter().map(|(s, _)| s).collect();
        let masks: Vec<ActionMask> = data.iter().map(|(_, m)| *m).collect();
        let res = model.infer(&states, &masks, &dev);
        assert_eq!(res.len(), 16);
        for (r, m) in res.iter().zip(&masks) {
            let sum: f32 = r.probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "tổng xác suất = {sum}");
            for a in 0..NUM_ACTIONS {
                if !m.is_index_valid(a) {
                    assert!(r.probs[a] < 1e-6, "hành động không hợp lệ {a} có p = {}", r.probs[a]);
                }
            }
            assert!(r.value.is_finite());
        }
    }

    #[test]
    fn padding_does_not_change_output() {
        // Cùng một trạng thái ngắn, suy luận riêng và trong batch với trạng thái dài hơn phải giống nhau.
        let dev = Device::flex();
        let model = small_cfg().init(&dev);
        let data = sample_states(40, 2);
        let short = data.iter().min_by_key(|(s, _)| s.len).unwrap();
        let long = data.iter().max_by_key(|(s, _)| s.len).unwrap();
        assert!(long.0.len > short.0.len);
        let alone = model.infer(&[&short.0], &[short.1], &dev)[0];
        let mixed = model.infer(&[&long.0, &short.0], &[long.1, short.1], &dev)[1];
        for a in 0..NUM_ACTIONS {
            assert!((alone.probs[a] - mixed.probs[a]).abs() < 1e-5);
        }
        assert!((alone.value - mixed.value).abs() < 1e-5);
    }

    #[test]
    fn learns_policy_and_value_targets() {
        // Kiểm tra backprop đi qua cả embedding, encoder và hai đầu: học thuộc mục tiêu cho 32 trạng thái.
        let dev = Device::flex().autodiff();
        let mut model = small_cfg().init(&dev);
        let mut optim = AdamConfig::new().init();
        let data = sample_states(32, 3);
        let states: Vec<&EncodedState> = data.iter().map(|(s, _)| s).collect();
        let masks: Vec<ActionMask> = data.iter().map(|(_, m)| *m).collect();
        // Mục tiêu: luôn chọn CheckCall (luôn hợp lệ); value = ±0.5 theo việc có phải button không.
        let target_v: Vec<f32> =
            data.iter().map(|(s, _)| if s.tokens[0].cont[5] > 0.5 { 0.5 } else { -0.5 }).collect();
        let target_v = Tensor::<1>::from_data(TensorData::new(target_v, [32]), &dev);
        let call_idx = Tensor::<2, Int>::from_data(
            TensorData::new(vec![Action::CheckCall.to_index() as i64; 32], [32, 1]),
            &dev,
        );
        let batch = StateBatch::new(&states, &masks, &dev);

        let mut first = None;
        let mut last = 0.0;
        for _ in 0..150 {
            let out = model.forward(&batch);
            let nll = out.log_probs.gather(1, call_idx.clone()).neg().mean();
            let mse = (out.value - target_v.clone()).powi_scalar(2).mean();
            let loss = nll + mse;
            last = loss.clone().into_scalar::<f32>();
            first.get_or_insert(last);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-3, model, grads);
        }
        let first = first.unwrap();
        assert!(last < first * 0.1, "loss không giảm đủ: {first} → {last}");

        let trained = model.valid();
        let res = trained.infer(&states, &masks, &Device::flex());
        assert!(res.iter().all(|r| r.probs[Action::CheckCall.to_index()] > 0.9));
    }

    #[test]
    fn save_load_roundtrip() {
        let dev = Device::flex();
        let cfg = small_cfg();
        let model = cfg.init(&dev);
        let data = sample_states(4, 4);
        let states: Vec<&EncodedState> = data.iter().map(|(s, _)| s).collect();
        let masks: Vec<ActionMask> = data.iter().map(|(_, m)| *m).collect();
        let before = model.infer(&states, &masks, &dev);

        let dir = std::env::temp_dir().join(format!("rl_model_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.bpk");
        model.save_file(&path).unwrap();
        let loaded = cfg.init(&dev).load_file(&path);
        let after = loaded.infer(&states, &masks, &dev);
        std::fs::remove_dir_all(&dir).ok();

        for (x, y) in before.iter().zip(&after) {
            assert_eq!(x.probs, y.probs);
            assert_eq!(x.value, y.value);
        }
    }
}

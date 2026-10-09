use crate::environment::action::NUM_ACTIONS;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// (inputs, targets, masks, weights)
pub type SampleBatch = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<bool>>, Vec<f32>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvantageSample {
    pub features: Vec<f32>,
    pub regrets: Vec<f32>, // size: NUM_ACTIONS
    pub mask: [bool; NUM_ACTIONS],
    pub iteration_weight: f32,
}

/// Bộ nhớ dùng RESERVOIR SAMPLING (đúng Deep CFR): mọi mẫu từng xuất hiện
/// đều có xác suất như nhau để nằm trong buffer, không bị mẫu mới đè mất mẫu cũ.
#[derive(Debug, Clone)]
pub struct AdvantageMemory {
    pub capacity: usize,
    pub buffer: Vec<AdvantageSample>,
    pub seen: usize,
}

impl AdvantageMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity.min(1_000_000)),
            seen: 0,
        }
    }

    pub fn push<R: Rng>(&mut self, sample: AdvantageSample, rng: &mut R) {
        self.seen += 1;
        if self.buffer.len() < self.capacity {
            self.buffer.push(sample);
        } else {
            let j = rng.gen_range(0..self.seen);
            if j < self.capacity {
                self.buffer[j] = sample;
            }
        }
    }

    pub fn sample_batch<R: Rng>(&self, rng: &mut R, batch_size: usize) -> SampleBatch {
        let size = batch_size.min(self.buffer.len());
        let mut inputs = Vec::with_capacity(size);
        let mut targets = Vec::with_capacity(size);
        let mut masks = Vec::with_capacity(size);
        let mut weights = Vec::with_capacity(size);

        for s in self.buffer.choose_multiple(rng, size) {
            inputs.push(s.features.clone());
            targets.push(s.regrets.clone());
            masks.push(s.mask.to_vec());
            weights.push(s.iteration_weight);
        }

        (inputs, targets, masks, weights)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

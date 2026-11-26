use crate::environment::action::NUM_ACTIONS;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// `(inputs, targets, masks, weights)`
pub type SampleBatch = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<bool>>, Vec<f32>);

pub trait WeightedSample {
    fn to_parts(&self) -> (&[f32], &[f32], &[bool; NUM_ACTIONS], f32);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvantageSample {
    pub features: Vec<f32>,
    pub regrets: Vec<f32>, // size: NUM_ACTIONS
    pub mask: [bool; NUM_ACTIONS],
    pub iteration_weight: f32,
}

impl WeightedSample for AdvantageSample {
    fn to_parts(&self) -> (&[f32], &[f32], &[bool; NUM_ACTIONS], f32) {
        (&self.features, &self.regrets, &self.mask, self.iteration_weight)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySample {
    pub features: Vec<f32>,
    pub action_probs: Vec<f32>, // size: NUM_ACTIONS
    pub mask: [bool; NUM_ACTIONS],
    pub iteration_weight: f32,
}

impl WeightedSample for StrategySample {
    fn to_parts(&self) -> (&[f32], &[f32], &[bool; NUM_ACTIONS], f32) {
        (&self.features, &self.action_probs, &self.mask, self.iteration_weight)
    }
}

/// Reservoir sampling buffer chuẩn Deep CFR (mọi mẫu từng xuất hiện có xác suất giữ lại ngang nhau).
#[derive(Debug, Clone)]
pub struct ReservoirMemory<T> {
    pub capacity: usize,
    pub buffer: Vec<T>,
    pub seen: usize,
}

impl<T> ReservoirMemory<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity.min(1_000_000)),
            seen: 0,
        }
    }

    pub fn push<R: Rng>(&mut self, sample: T, rng: &mut R) {
        self.seen += 1;
        if self.buffer.len() < self.capacity {
            self.buffer.push(sample);
        } else if self.capacity > 0 {
            let j = rng.gen_range(0..self.seen);
            if j < self.capacity {
                self.buffer[j] = sample;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl<T: WeightedSample> ReservoirMemory<T> {
    pub fn sample_batch<R: Rng>(&self, rng: &mut R, batch_size: usize) -> SampleBatch {
        let size = batch_size.min(self.buffer.len());
        let mut inputs = Vec::with_capacity(size);
        let mut targets = Vec::with_capacity(size);
        let mut masks = Vec::with_capacity(size);
        let mut weights = Vec::with_capacity(size);

        for s in self.buffer.choose_multiple(rng, size) {
            let (feat, tgt, m, w) = s.to_parts();
            inputs.push(feat.to_vec());
            targets.push(tgt.to_vec());
            masks.push(m.to_vec());
            weights.push(w);
        }

        (inputs, targets, masks, weights)
    }
}

pub type AdvantageMemory = ReservoirMemory<AdvantageSample>;
pub type StrategyMemory = ReservoirMemory<StrategySample>;

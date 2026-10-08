use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySample {
    pub features: Vec<f32>,
    pub action_probs: Vec<f32>, // size: NUM_ACTIONS (6)
    pub iteration_weight: f32,
}

#[derive(Debug, Clone)]
pub struct StrategyMemory {
    pub capacity: usize,
    pub buffer: Vec<StrategySample>,
    pub insert_idx: usize,
}

impl StrategyMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
            insert_idx: 0,
        }
    }

    pub fn push(&mut self, sample: StrategySample) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(sample);
        } else {
            self.buffer[self.insert_idx] = sample;
            self.insert_idx = (self.insert_idx + 1) % self.capacity;
        }
    }

    pub fn sample_batch<R: Rng>(&self, rng: &mut R, batch_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n = self.buffer.len();
        let size = batch_size.min(n);
        let mut inputs = Vec::with_capacity(size);
        let mut targets = Vec::with_capacity(size);

        let chosen: Vec<&StrategySample> = self.buffer.choose_multiple(rng, size).collect();
        for sample in chosen {
            inputs.push(sample.features.clone());
            targets.push(sample.action_probs.clone());
        }

        (inputs, targets)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

//! Bước 0 của bot RL: kiểm tra `burn` chạy được trên GPU (Vulkan / Iris Xe, hoặc CUDA).
//!
//!   cargo run --release --features rl --bin rl_gpu_check
//!   cargo run --release --features rl-cuda --bin rl_gpu_check   (máy có GPU NVIDIA)
//!
//! 1. Nhân ma trận trên CPU (flex) và GPU, so kết quả và tốc độ.
//! 2. Huấn luyện thử một lớp Linear bằng autodiff + Adam trên GPU (học y = 3x₀ − 2x₁ + 1).
//! 3. Đo tốc độ transformer thật (suy luận + một bước train) trên CPU và GPU.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, DeviceKind, Distribution, Tensor};
use poker_tournament::environment::{ActionMask, Environment};
use poker_tournament::rl::encoding::{encode, EncodedState};
use poker_tournament::rl::model::{PokerTransformer, PokerTransformerConfig, StateBatch};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn gpu_device() -> Device {
    #[cfg(feature = "rl-cuda")]
    {
        return Device::cuda(0);
    }
    #[allow(unreachable_code)]
    Device::vulkan(DeviceKind::DefaultDevice)
}

fn bench_matmul(device: &Device, n: usize, reps: usize) -> (Vec<f32>, f64) {
    let a = Tensor::<2>::from_data(
        burn::tensor::TensorData::new((0..n * n).map(|i| ((i % 97) as f32 - 48.0) / 97.0).collect::<Vec<_>>(), [n, n]),
        device,
    );
    let b = Tensor::<2>::from_data(
        burn::tensor::TensorData::new((0..n * n).map(|i| ((i % 89) as f32 - 44.0) / 89.0).collect::<Vec<_>>(), [n, n]),
        device,
    );
    // khởi động (biên dịch kernel)
    let _ = a.clone().matmul(b.clone()).into_data();
    let t0 = Instant::now();
    let mut out = a.clone().matmul(b.clone());
    for _ in 1..reps {
        out = a.clone().matmul(b.clone()) + out * 0.0;
    }
    let data = out.into_data().try_to_vec::<f32>().unwrap();
    let secs = t0.elapsed().as_secs_f64();
    let gflops = 2.0 * (n as f64).powi(3) * reps as f64 / secs / 1e9;
    (data, gflops)
}

fn main() {
    let cpu = Device::flex();
    let gpu = gpu_device();
    println!("GPU device: {:?}", gpu);

    // 1. Nhân ma trận
    let n = 1024;
    let (c_cpu, gf_cpu) = bench_matmul(&cpu, n, 3);
    let (c_gpu, gf_gpu) = bench_matmul(&gpu, n, 20);
    let max_err = c_cpu.iter().zip(&c_gpu).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("matmul {n}x{n}: CPU {gf_cpu:.1} GFLOP/s | GPU {gf_gpu:.1} GFLOP/s | sai lệch tối đa {max_err:.2e}");
    assert!(max_err < 1e-2, "Kết quả GPU lệch CPU quá nhiều");

    // 2. Autodiff + Adam
    let dev = gpu.clone().autodiff();
    let mut model: Linear = LinearConfig::new(2, 1).init(&dev);
    let mut optim = AdamConfig::new().init();
    let t0 = Instant::now();
    let mut loss_val = f32::NAN;
    for step in 0..300 {
        let x = Tensor::<2>::random([256, 2], Distribution::Uniform(-1.0, 1.0), &dev);
        let w = Tensor::<2>::from_floats([[3.0f32], [-2.0]], &dev);
        let y = x.clone().matmul(w) + 1.0;
        let pred = model.forward(x);
        let loss = (pred - y).powi_scalar(2).mean();
        loss_val = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(0.05, model, grads);
        if step % 100 == 0 {
            println!("  step {step:>3}: loss {loss_val:.5}");
        }
    }
    let params: Vec<f32> = model.weight.val().into_data().try_to_vec::<f32>().unwrap();
    let bias: Vec<f32> = model.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap();
    println!(
        "Autodiff: loss cuối {loss_val:.5}, w = {:?}, b = {:?} (mong đợi [3, -2], 1) trong {:.2}s",
        params,
        bias,
        t0.elapsed().as_secs_f64()
    );
    assert!(loss_val < 1e-2, "Autodiff/Adam không hội tụ trên GPU");
    let _ = model.num_params();
    println!("✅ Bước 0 OK: burn chạy được trên GPU.");

    // 3. Tốc độ transformer thật
    let samples = sample_states(512);
    for (name, d) in [("CPU", cpu.clone()), ("GPU", gpu.clone())] {
        bench_transformer(name, &d, &samples);
    }
}

fn sample_states(n: usize) -> Vec<(EncodedState, ActionMask)> {
    let mut rng = StdRng::seed_from_u64(7);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut env = Environment::new(2, &[10_000, 10_000], 100, 50, 0);
        env.reset_hand(&mut rng, 0);
        loop {
            let mask = env.get_action_mask();
            out.push((encode(&env.state, env.state.current_player_idx), mask));
            let valid = mask.valid_actions();
            if env.step(valid[rng.gen_range(0..valid.len())]).unwrap() || out.len() >= n {
                break;
            }
        }
    }
    out
}

fn bench_transformer(name: &str, device: &Device, samples: &[(EncodedState, ActionMask)]) {
    let cfg = PokerTransformerConfig::new();
    let model = cfg.init(device);
    let states: Vec<&EncodedState> = samples.iter().map(|(s, _)| s).collect();
    let masks: Vec<ActionMask> = samples.iter().map(|(_, m)| *m).collect();
    for &bs in &[64usize, 512] {
        let _ = model.infer(&states[..bs], &masks[..bs], device); // khởi động
        let reps = 5;
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = model.infer(&states[..bs], &masks[..bs], device);
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "[{name}] transformer {} tham số | suy luận batch {bs}: {:.1} ms ({:.0} trạng thái/s)",
            model.num_params(),
            per * 1e3,
            bs as f64 / per
        );
    }

    // Một bước train (forward + backward + Adam) với batch 512.
    let ad = device.clone().autodiff();
    let mut m = cfg.init(&ad);
    let mut optim = AdamConfig::new().init();
    let batch = StateBatch::new(&states, &masks, &ad);
    let mut step = |m: PokerTransformer| {
        let out = m.forward(&batch);
        let loss = out.log_probs.exp().mean() + out.value.powi_scalar(2).mean();
        let _ = loss.clone().into_scalar::<f32>(); // đồng bộ thiết bị (như khi log loss lúc train)
        let grads = GradientsParams::from_grads(loss.backward(), &m);
        optim.step(1e-4, m, grads)
    };
    m = step(m);
    let reps = 3;
    let t0 = Instant::now();
    for _ in 0..reps {
        m = step(m);
    }
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    println!("[{name}] một bước train batch 512: {:.0} ms ({:.0} mẫu/s)", per * 1e3, 512.0 / per);
}

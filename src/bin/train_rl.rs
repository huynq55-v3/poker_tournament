//! Bot RL heads-up 100 BB: tự chơi để thu thập dữ liệu (bước 3). PPO sẽ được thêm ở bước 4.
//!
//!   cargo run --release --features rl --bin train_rl
//!   RL_DEVICE=cpu cargo run --release --features rl --bin train_rl
//!
//! Biến môi trường: RL_HANDS (mặc định 4096), RL_PARALLEL (mặc định 512).

use poker_tournament::environment::Action;
use poker_tournament::rl::device_from_env;
use poker_tournament::rl::model::PokerTransformerConfig;
use poker_tournament::rl::selfplay::{collect, evaluate, Agent, Opponent, RolloutConfig};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let device = device_from_env();
    let hands = env_usize("RL_HANDS", 4096);
    let parallel = env_usize("RL_PARALLEL", 512);
    println!("Thiết bị: {device:?} | {hands} ván | {parallel} bàn song song");

    let model = PokerTransformerConfig::new().init(&device);

    // Khởi động (biên dịch kernel GPU cho các độ dài chuỗi) rồi mới đo.
    let _ = collect(&model, Opponent::SelfPlay, &RolloutConfig { hands: parallel, parallel, seed: 99 }, &device);

    let cfg = RolloutConfig { hands, parallel, seed: 1 };
    let batch = collect(&model, Opponent::SelfPlay, &cfg, &device);
    let st = &batch.stats;
    println!(
        "Tự chơi: {} ván, {} quyết định ({:.2}/ván) trong {:.1}s → {:.0} ván/s, {:.0} quyết định/s",
        st.hands,
        st.decisions,
        st.decisions as f64 / st.hands as f64,
        st.seconds,
        st.hands as f64 / st.seconds,
        st.decisions as f64 / st.seconds
    );
    let freq: Vec<String> = Action::ALL
        .iter()
        .map(|a| format!("{} {:.1}%", a.name(), 100.0 * st.action_counts[a.to_index()] as f64 / st.decisions as f64))
        .collect();
    println!("Tần suất hành động: {}", freq.join(" | "));
    println!("Số quỹ đạo: {}", batch.trajectories.len());

    // Mốc ban đầu: mô hình chưa train phải thua bot Heuristic.
    for (name, opp) in [("Random", Agent::Random), ("Heuristic", Agent::Heuristic)] {
        let r = evaluate(Agent::Model(&model), opp, 2000, parallel, 7, &device);
        println!("Mô hình chưa train vs {name}: {:+.1} ± {:.1} BB/100", r.bb_per_100, r.se_bb_per_100);
    }
}

//! 🚀 DEEP CFR CASH GAME TRAINER (6-MAX REAL COIN / DEEP STACK 40-250 BB / 5.5% RAKE)
//!
//! Tối ưu hóa chuyên dụng cho Cash Game PokerStars:
//! - Bàn 6-Max tiêu chuẩn.
//! - Thuần Chip-EV (không dùng Tournament ICM, mỗi 1 chip = tiền thật).
//! - Mô phỏng độ sâu chip bất đối xứng (40 BB đến 250 BB) giải quyết triệt để luật Anti-Ratholing ("kiếm được tiền không rút ra được").
//! - Khấu trừ 5.5% Rake sàn thực tế khi có Flop ("No Flop, No Drop").
//! - Tách biệt độc lập: Lưu ra deep_cfr_cash_model.json, không ảnh hưởng đến mô hình Tournament PokerNow.

use poker_tournament::benchmark::{
    evaluate_against_champion_cash, ArenaMatchResult,
};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::FEATURE_DIM;
use poker_tournament::neural_network::MLP;
use rand::thread_rng;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const BEST_ADV: &str = "deep_cfr_cash_model.json";
const BEST_STRAT: &str = "deep_cfr_cash_strategy.json";
const LAST_ADV: &str = "deep_cfr_cash_last_model.json";
const LAST_STRAT: &str = "deep_cfr_cash_last_strategy.json";
const BENCH_CSV: &str = "benchmark_cash_log.csv";

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("============================================================");
        println!("🚀 POKER AI DEEP CFR - CASH GAME 6-MAX TRAINER 🚀");
        println!("============================================================");
        println!("Cú pháp: cargo run --release --bin train_cash -- [TÙY CHỌN]");
        println!();
        println!("Tùy chọn:");
        println!("  --iters <N>           Số iteration (mặc định: 10000)");
        println!("  --traversals <N>      Số ván duyệt cây mỗi iter (mặc định: 250)");
        println!("  --train-steps <N>     Số bước SGD train mạng mỗi iter (mặc định: 400)");
        println!("  --bench-every <N>     Tần suất đấu trường Arena (mặc định: 10)");
        println!("  --arena-hands <N>     Số ván đấu trường Arena (mặc định: 6000)");
        println!("  --from <FILE>         Load checkpoint Advantage Net (nếu có)");
        println!("  --from-strat <FILE>   Load checkpoint Strategy Net (nếu có)");
        println!("============================================================");
        return;
    }

    let iters: usize = arg(&args, "--iters", 10_000);
    let traversals: usize = arg(&args, "--traversals", 250);
    let train_steps: usize = arg(&args, "--train-steps", 400);
    let bench_every: usize = arg(&args, "--bench-every", 10);
    let arena_hands: usize = arg(&args, "--arena-hands", 6000);
    let batch_size: usize = 512;

    println!("============================================================");
    println!("🔥 KHỞI ĐỘNG DEEP CFR CASH GAME TRAINER (POKERSTARS REAL COIN) 🔥");
    println!("============================================================");
    println!("  • Quy mô bàn: 6-Max Cash Game");
    println!("  • Cơ chế thưởng: Thuần Chip-EV (Mỗi chip = Tiền thật)");
    println!("  • Dải Stack: 40 BB -> 250 BB (Mô phỏng tích lũy tiền Anti-Ratholing)");
    println!("  • Phí sàn (Rake): 5.5% (No Flop No Drop, Cap 3 BB)");
    println!("  • File Output: {} & {}", BEST_ADV, BEST_STRAT);
    println!("  • Feature Scaling: Cơ số 150 BB (Dải mượt 0..300 BB không kịch trần)");
    println!("============================================================");

    let hidden_dims = [256, 256, 128];
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, 2_500_000, false);
    trainer.is_cash_game = true;
    trainer.max_players = 6;
    trainer.fixed_players = Some(6);
    trainer.use_icm = false;
    trainer.stack_bb = 0; // Kích hoạt sample_cash_game_stacks (40 - 250 BB)
    trainer.lr = 0.0005;

    // Load checkpoint nếu có
    let from_adv = args.iter().position(|a| a == "--from")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or(BEST_ADV);
    let from_strat = args.iter().position(|a| a == "--from-strat")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or(BEST_STRAT);

    if std::path::Path::new(from_adv).exists() {
        println!("📥 Đang nạp checkpoint có sẵn: {}", from_adv);
        let _ = trainer.load_models(from_adv, from_strat);
    } else {
        println!("🌱 Khởi tạo mạng mới hoàn toàn cho Cash Game (Zero-bias Deep Stack).");
    }

    let mut current_champion_net: MLP = trainer.advantage_net.read().unwrap().clone();
    let _ = trainer.save_models(BEST_ADV, BEST_STRAT);

    let mut rng = thread_rng();
    let train_start = Instant::now();

    for iter in 1..=iters {
        let it0 = Instant::now();
        let adv_loss = trainer.step_iteration(traversals, train_steps, batch_size, &mut rng);
        println!(
            "   📍 Cash Iter {:05}/{} | Adv: {:7} | Strat: {:7} | Loss: {:.5} | {:.1}s",
            iter, iters,
            trainer.adv_memory.len(), trainer.strategy_memory.len(),
            adv_loss, it0.elapsed().as_secs_f64()
        );

        if iter % bench_every == 0 {
            let bt = Instant::now();
            println!("\n   ⚔️  ĐẤU TRƯỜNG CASH 6-MAX (40-250 BB, 5.5% Rake): Challenger (Iter {}) vs Champion...", iter);
            let challenger_policy = DeepCFRPolicy::new(Arc::clone(&trainer.advantage_net), false);
            let champion_policy = DeepCFRPolicy::new(Arc::new(RwLock::new(current_champion_net.clone())), false);

            let arena_res: ArenaMatchResult = evaluate_against_champion_cash(
                &challenger_policy,
                &champion_policy,
                arena_hands,
                6,
            );

            println!(
                "   📊 Kết quả Arena Cash: {:+.2} ± {:.2} bb/100 | TrashShove: {:.2}% | {} ván trong {:.1}s",
                arena_res.challenger_bb_per_100,
                arena_res.ci95,
                arena_res.challenger_trash_shove_pct,
                arena_res.hands,
                bt.elapsed().as_secs_f64()
            );

            if arena_res.is_winner {
                println!(
                    "   👑 SOÁN NGÔI CHAMPION! Iter {} thắng vượt trội {:+.2} bb/100 -> Cập nhật Best Model!",
                    iter, arena_res.challenger_bb_per_100
                );
                current_champion_net = trainer.advantage_net.read().unwrap().clone();
                let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
            } else {
                println!("   🛡️ Champion vẫn giữ vững vị thế.");
            }

            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(BENCH_CSV) {
                let _ = writeln!(f, "{},cash_arena,{:.2},{:.2},{:.2},{}", 
                    iter, 
                    arena_res.challenger_bb_per_100, 
                    arena_res.ci95, 
                    arena_res.challenger_trash_shove_pct, 
                    arena_res.hands
                );
            }

            // Luôn lưu checkpoint gần nhất
            let _ = trainer.save_models(LAST_ADV, LAST_STRAT);
            println!();
        }
    }

    println!("🎉 Hoàn thành huấn luyện Cash Game trong {:.1} phút!", train_start.elapsed().as_secs_f64() / 60.0);
}

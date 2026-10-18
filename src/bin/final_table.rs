//! 🚀 DEEP CFR TOURNAMENT FINAL TABLE TRAINER (8-MAX / ASYMMETRIC RANDOM STACK / PURE CHIP-EV MAX)
//!
//! Tối ưu hóa chuyên biệt cho Bàn Chung Kết (Final Table):
//! - Bàn tối đa 8 người (8-Max Final Table).
//! - Tiêu chí thuần Chip-EV Max (Không dùng Tournament ICM, Không trừ Rake sàn).
//! - Phân phối Stack ngẫu nhiên bất đối xứng (Dirichlet 4 BB đến 150 BB) mô phỏng Chip Leader, Mid Stack và Short Stack thực chiến.
//! - Tách biệt độc lập: Lưu ra deep_cfr_final_table_model.json.

use poker_tournament::benchmark::{
    evaluate_against_champion_final_table, ArenaMatchResult,
};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::FEATURE_DIM;
use poker_tournament::neural_network::MLP;
use rand::thread_rng;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const BEST_ADV: &str = "deep_cfr_final_table_model.json";
const BEST_STRAT: &str = "deep_cfr_final_table_strategy.json";
const LAST_ADV: &str = "deep_cfr_final_table_last_model.json";
const LAST_STRAT: &str = "deep_cfr_final_table_last_strategy.json";
const BENCH_CSV: &str = "benchmark_final_table_log.csv";

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
        println!("🚀 POKER AI DEEP CFR - FINAL TABLE (8-MAX / MAX EV) TRAINER 🚀");
        println!("============================================================");
        println!("Cú pháp: cargo run --release --bin final_table -- [TÙY CHỌN]");
        println!();
        println!("Tùy chọn:");
        println!("  --iters <N>           Số iteration (mặc định: 10000)");
        println!("  --traversals <N>      Số ván duyệt cây mỗi iter (mặc định: 250)");
        println!("  --train-steps <N>     Số bước SGD train mạng mỗi iter (mặc định: 400)");
        println!("  --bench-every <N>     Tần suất đấu trường Arena (mặc định: 10)");
        println!("  --arena-hands <N>     Số ván đấu trường Arena (mặc định: 6000)");
        println!("  --max-players <N>     Số người tối đa trên bàn (mặc định: 8)");
        println!("  --fixed-players <N>   Cố định số người (mặc định: 8, đặt 0 để ngẫu nhiên 2..=8)");
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
    let max_players: usize = arg(&args, "--max-players", 8);
    let fixed_players: usize = arg(&args, "--fixed-players", 8);
    let batch_size: usize = 512;

    println!("============================================================");
    println!("🔥 KHỞI ĐỘNG DEEP CFR FINAL TABLE TRAINER (8-MAX RANDOM STACK) 🔥");
    println!("============================================================");
    println!("  • Quy mô bàn: {}-Max Final Table Tournament", max_players);
    println!("  • Cố định người chơi: {}", if fixed_players > 0 { format!("{} người cố định", fixed_players) } else { "Ngẫu nhiên 2..=8 (mô phỏng loại người)".to_string() });
    println!("  • Cơ chế thưởng: Thuần Chip-EV Max (Không ICM, Không phế Rake sàn)");
    println!("  • Phân phối Stack: Random bất đối xứng (Dirichlet 4 BB -> 150 BB)");
    println!("  • File Output: {} & {}", BEST_ADV, BEST_STRAT);
    println!("  • Feature Scaling: Cơ số 150 BB (Dải mượt 0..300 BB không kịch trần)");
    println!("============================================================");

    let hidden_dims = [256, 256, 128];
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, 2_500_000, false);
    trainer.is_cash_game = false; // Thuần Tournament Chip-EV (không trừ 5.5% Rake sàn)
    trainer.max_players = max_players;
    trainer.fixed_players = if fixed_players > 0 { Some(fixed_players) } else { None };
    trainer.use_icm = false; // Thuần Chip-EV Max theo yêu cầu
    trainer.stack_bb = 0; // Kích hoạt random stack (Dirichlet 4-150 BB + equal depths)
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
        println!("🌱 Khởi tạo mạng mới hoàn toàn cho Final Table (Max EV / Random Stack).");
    }

    let mut current_champion_net: MLP = trainer.advantage_net.read().unwrap().clone();
    let _ = trainer.save_models(BEST_ADV, BEST_STRAT);

    let mut rng = thread_rng();
    let train_start = Instant::now();

    for iter in 1..=iters {
        let it0 = Instant::now();
        let adv_loss = trainer.step_iteration(traversals, train_steps, batch_size, &mut rng);
        println!(
            "   📍 FinalTable Iter {:05}/{} | Adv: {:7} | Strat: {:7} | Loss: {:.5} | {:.1}s",
            iter, iters,
            trainer.adv_memory.len(), trainer.strategy_memory.len(),
            adv_loss, it0.elapsed().as_secs_f64()
        );

        if iter % bench_every == 0 {
            let bt = Instant::now();
            println!("\n   ⚔️  ĐẤU TRƯỜNG FINAL TABLE {}-MAX (Random Stack, Pure Chip-EV): Challenger (Iter {}) vs Champion...", max_players, iter);
            let challenger_policy = DeepCFRPolicy::new(Arc::clone(&trainer.advantage_net), false);
            let champion_policy = DeepCFRPolicy::new(Arc::new(RwLock::new(current_champion_net.clone())), false);

            let arena_res: ArenaMatchResult = evaluate_against_champion_final_table(
                &challenger_policy,
                &champion_policy,
                arena_hands,
                max_players,
            );

            println!(
                "   📊 Kết quả Arena Final Table: {:+.2} ± {:.2} bb/100 | TrashShove: {:.2}% | {} ván trong {:.1}s",
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
                let _ = writeln!(f, "{},final_table_arena,{:.2},{:.2},{:.2},{}", 
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

    println!("🎉 Hoàn thành huấn luyện Final Table trong {:.1} phút!", train_start.elapsed().as_secs_f64() / 60.0);
}

use poker_tournament::benchmark::{
    append_csv, evaluate_against_champion_table, evaluate_against_champion_tournament,
    print_table, run_suite, score, simulate_demo_tournament, simulate_hero_tournament,
    TournamentArenaResult,
};
use poker_tournament::deep_cfr::DeepCFRPolicy;
use poker_tournament::deep_cfr::DeepCFRTrainer;
use poker_tournament::environment::FEATURE_DIM;
use poker_tournament::neural_network::MLP;
use rand::thread_rng;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const BEST_ADV: &str = "deep_cfr_model.json";
const BEST_STRAT: &str = "deep_cfr_strategy.json";
const LAST_ADV: &str = "deep_cfr_last_model.json";
const LAST_STRAT: &str = "deep_cfr_last_strategy.json";
const BENCH_CSV: &str = "benchmark_log.csv";

/// Phân tích tham số dòng lệnh linh hoạt:
/// - Mặc định: Bàn 8-Max, Tournament ICM, 50 BB khởi điểm, giả lập đến khi hết tour
fn parse_args() -> (usize, bool) {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("============================================================");
        println!("🚀 POKER AI DEEP CFR - TOURNAMENT 8-MAX (50 BB / HẾT TOUR) 🚀");
        println!("============================================================");
        println!("Cú pháp: cargo run --release -- [SỐ_NGƯỜI: 6|8] [CHẾ_ĐỘ: icm|ev]");
        println!("Hoặc:   cargo run --release -- [--players 6|8] [--icm|--ev]");
        println!();
        println!("Tham số 1: Quy mô bàn");
        println!("  8 hoặc --players 8 hoặc --8max : Bàn 8-Max (Mặc định cho giải đấu)");
        println!("  6 hoặc --players 6 hoặc --6max : Bàn 6-Max");
        println!();
        println!("Tham số 2: Hàm thưởng (Reward)");
        println!("  icm hoặc --icm                 : Bật Tournament ICM (Sit & Go, Payout Zero-Sum)");
        println!("  ev hoặc --ev hoặc --chip-ev    : Dùng Chip-EV thuần (Cash Game / EV Max)");
        println!();
        println!("Độ sâu stack & Cơ chế giải đấu:");
        println!("  🟢 Mỗi người chơi ban đầu có 50 BB (Tổng bàn 8-Max là 400 BB).");
        println!("  🏆 Giả lập giải đấu toàn diện từ 8 người rụng dần đến khi chỉ còn 1 người vô địch (Hết tour).");
        println!();
        println!("Ví dụ thực chiến:");
        println!("  1. Bàn 8-max Tournament ICM (Khuyên dùng):");
        println!("     cargo run --release");
        println!("  2. Bàn 6-max, dùng Tournament ICM:");
        println!("     cargo run --release -- 6 icm");
        println!("  3. Bàn 8-max, dùng EV thuần:");
        println!("     cargo run --release -- 8 ev");
        println!("============================================================");
        std::process::exit(0);
    }

    let mut max_players = 8usize;
    let mut custom_players = false;

    // 1. Quét flags cho số người chơi
    for (i, a) in args.iter().enumerate() {
        if a == "--players" || a == "--max-players" || a == "-p" {
            if let Some(val) = args.get(i + 1) {
                if let Ok(p) = val.parse::<usize>() {
                    max_players = p;
                    custom_players = true;
                }
            }
        } else if a == "--6max" || a == "-6" || a == "6max" || a == "6-max" {
            max_players = 6;
            custom_players = true;
        } else if a == "--8max" || a == "-8" || a == "8max" || a == "8-max" {
            max_players = 8;
            custom_players = true;
        }
    }

    // 2. Quét flags cho chế độ ICM / EV thuần
    let mut use_icm = true;
    let mut custom_mode = false;

    for a in &args {
        let lower = a.to_lowercase();
        if lower == "--icm" || lower == "icm" {
            use_icm = true;
            custom_mode = true;
        } else if lower == "--ev" || lower == "ev" || lower == "--chip-ev" || lower == "chip-ev" || lower == "chipev" || lower == "--no-icm" {
            use_icm = false;
            custom_mode = true;
        }
    }

    // 3. Quét positional args (nếu chưa chỉ định qua flags)
    let positional: Vec<&String> = args
        .iter()
        .filter(|a| !a.starts_with('-'))
        .collect();

    if !custom_players && !positional.is_empty() {
        if let Ok(p) = positional[0].parse::<usize>() {
            if p == 6 || p == 8 {
                max_players = p;
            }
        }
    }

    if !custom_mode && positional.len() >= 2 {
        let mode_str = positional[1].to_lowercase();
        if mode_str == "icm" {
            use_icm = true;
        } else if mode_str == "ev" || mode_str == "chip-ev" || mode_str == "chipev" {
            use_icm = false;
        }
    }

    if max_players != 6 && max_players != 8 {
        println!("⚠️  Cảnh báo: Chỉ hỗ trợ bàn 6-max hoặc 8-max. Tự động áp dụng bàn {}-max.", if max_players <= 6 { 6 } else { 8 });
        max_players = if max_players <= 6 { 6 } else { 8 };
    }

    (max_players, use_icm)
}

fn main() {
    let (max_players, use_icm) = parse_args();
    let initial_bb = 50u32; // 🟢 Mỗi người chơi ban đầu có 50 BB (tổng 400 BB cả giải)
    let mode_desc = if use_icm { "TOURNAMENT ICM (FULL RUN TO WINNER)" } else { "CHIP-EV THUẦN" };

    println!("============================================================");
    println!("🚀 POKER AI DEEP CFR ({}-MAX TOURNAMENT / 50 BB KHỞI ĐIỂM) 🚀", max_players);
    println!("⚙️  CẤU HÌNH: Bàn {}-Max | Chế độ: {} | Stack: {} BB/người", max_players, mode_desc, initial_bb);
    println!("🏆 MỤC TIÊU: Giả lập giải đấu toàn diện từ {} người rụng dần đến khi hết tour!", max_players);
    println!("============================================================");

    let mut rng = rand::thread_rng();

    println!("\n[0] 📐 Preparing preflop equity table...");
    let t0 = Instant::now();
    let _ = poker_tournament::equity::preflop_table();
    println!("   ✅ Ready in {:.1}s", t0.elapsed().as_secs_f64());

    println!("\n[1] 🏋️ Deep CFR training with {}-Max Tournament Gating (50 BB Stack -> End of Tour)...", max_players);
    
    // Kiến trúc mạng nơ-ron: 3 tầng ẩn đủ sâu cho 177 features
    let hidden_dims = [256, 256, 256];
    
    // Capacity 5M mẫu vừa vặn bộ nhớ RAM
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, 5_000_000, use_icm);
    trainer.max_players = max_players;   // 🟢 Bàn max 8 người (có phủ Heads-up, 3-max, Bubble 5-max)
    trainer.fixed_players = None;        // Phủ đều mọi giai đoạn giải đấu
    trainer.use_icm = use_icm;           // 🟢 Bật Tournament ICM (Malmuth-Harville)
    trainer.stack_bb = initial_bb;       // 🟢 Mỗi người ban đầu 50 BB (tổng chip toàn giải bảo toàn 400 BB)
    trainer.is_cash_game = false;        // Giải đấu Tournament (No rake)
    trainer.lr = 0.0005;

    let total_iterations = 10_000;       // Chạy liên tục 10,000 iterations
    let traversals_per_iter = 250;       // Duyệt cây mỗi iter
    let train_steps_per_iter = 400;      // Ép mạng học kỹ các tình huống
    let batch_size = 512;                // Kích thước vàng cho SIMD/Rayon

    let bench_every = 10;                // Đấu trường Arena mỗi 10 iters
    let arena_tournaments = 100;         // Đấu 100 giải đấu Sit & Go trọn vẹn mỗi lần kiểm tra Champion
    let arena_hands_cash = 6000;         // Nếu ở chế độ Cash EV
    let bench_hands = 6000;              // Đánh giá đối thủ tĩnh
    let depths = [50u32, 25u32, 12u32, 8u32]; // Các mốc stack tiêu biểu trong Tournament 50 BB

    let _ = std::fs::remove_file(BENCH_CSV);

    if std::path::Path::new(BEST_ADV).exists() {
        println!("📥 Đang nạp checkpoint Champion có sẵn: {}", BEST_ADV);
        match trainer.load_models(BEST_ADV, BEST_STRAT) {
            Ok(true) => println!("   ✅ Đã nạp thành công Advantage Net và Strategy Net!"),
            Ok(false) => println!("   ✅ Đã nạp Advantage Net (Strategy Net khởi tạo mới)!"),
            Err(e) => println!("   ⚠️  Không thể nạp checkpoint ({}), tiếp tục khởi tạo mới.", e),
        }
    } else {
        println!("🌱 Khởi tạo mạng mới hoàn toàn cho Tournament 8-Max.");
        let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
    }

    let mut current_champion_net: MLP = trainer.advantage_net.read().unwrap().clone();

    let train_start = Instant::now();
    for iter in 1..=total_iterations {
        let it0 = Instant::now();
        let adv_loss = trainer.step_iteration(traversals_per_iter, train_steps_per_iter, batch_size, &mut rng);
        println!(
            "   📍 Iter {:05}/{} | Adv: {:7} | Strat: {:7} | Loss: {:.5} | {:.1}s",
            iter, total_iterations,
            trainer.adv_memory.len(), trainer.strategy_memory.len(),
            adv_loss, it0.elapsed().as_secs_f64()
        );

        if iter % bench_every == 0 {
            let bt = Instant::now();
            let challenger_policy = DeepCFRPolicy::new(Arc::clone(&trainer.advantage_net), false);
            let champion_policy = DeepCFRPolicy::new(Arc::new(RwLock::new(current_champion_net.clone())), false);

            if use_icm {
                println!("\n   ⚔️  ĐẤU TRƯỜNG TOURNAMENT {}-MAX (50 BB Khởi Điểm - Đánh Đến Khi Hết Tour): Challenger (Iter {}) vs Champion...", max_players, iter);
                let tour_res: TournamentArenaResult = evaluate_against_champion_tournament(
                    &challenger_policy,
                    &champion_policy,
                    arena_tournaments,
                    initial_bb,
                    max_players,
                );

                println!(
                    "   📊 Kết quả Tournament Arena: Vô địch: {} vs {} (Champ) | ITM (Top 3): {} | Rank TB: {:.2} vs {:.2} | ICM EV: {:+.3} | TrashShove: {:.2}% | {} ván ({:.1} ván/tour) trong {:.1}s",
                    tour_res.challenger_first_places,
                    tour_res.champion_first_places,
                    tour_res.challenger_itm_count,
                    tour_res.challenger_avg_rank,
                    tour_res.champion_avg_rank,
                    tour_res.challenger_avg_icm_payout,
                    tour_res.challenger_trash_shove_pct,
                    tour_res.total_hands,
                    tour_res.avg_hands_per_tour,
                    bt.elapsed().as_secs_f64()
                );

                if tour_res.is_winner {
                    println!(
                        "   👑 SOÁN NGÔI THÀNH CÔNG! Challenger vượt trội tại giải đấu (Rank TB {:.2} vs {:.2}) -> Đã lưu Champion mới vào {}!",
                        tour_res.challenger_avg_rank, tour_res.champion_avg_rank, BEST_ADV
                    );
                    current_champion_net = trainer.advantage_net.read().unwrap().clone();
                    let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
                } else {
                    println!("   🛡️  Champion giữ vững ngôi vương. Chưa đủ cách biệt thống kê (hoặc TrashShove > 5%).");
                }
            } else {
                println!("\n   ⚔️  ĐẤU TRƯỜNG {}-MAX (50 BB): Challenger (Iter {}) vs Champion...", max_players, iter);
                let arena_res = evaluate_against_champion_table(
                    &challenger_policy,
                    &champion_policy,
                    arena_hands_cash,
                    initial_bb,
                    max_players,
                );

                println!(
                    "   📊 Kết quả Arena: {:+.2} ± {:.2} bb/100 | TrashShove: {:.2}% | {} ván trong {:.1}s",
                    arena_res.challenger_bb_per_100,
                    arena_res.ci95,
                    arena_res.challenger_trash_shove_pct,
                    arena_res.hands,
                    bt.elapsed().as_secs_f64()
                );

                if arena_res.is_winner {
                    println!(
                        "   👑 SOÁN NGÔI THÀNH CÔNG! (+{:.2} bb/100) -> Đã lưu Champion mới vào {}!",
                        arena_res.challenger_bb_per_100, BEST_ADV
                    );
                    current_champion_net = trainer.advantage_net.read().unwrap().clone();
                    let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
                } else {
                    println!("   🛡️  Champion giữ vững ngôi vương. Chưa đủ cách biệt thống kê (hoặc TrashShove > 5%).");
                }
            }

            let r_adv = run_suite(&challenger_policy, bench_hands, &depths);
            print_table("Tham khảo đối thủ tĩnh", &r_adv);
            let _ = append_csv(BENCH_CSV, iter, "adv", &r_adv);

            let s = score(&r_adv);
            println!("   🧪 Benchmark log score: {:+.1} bb/100\n", s);

            // Tự động lưu checkpoint mới nhất
            let _ = trainer.save_models(LAST_ADV, LAST_STRAT);
        }

        if iter % 100 == 0 {
            let _ = std::fs::create_dir_all("checkpoints");
            let a = format!("checkpoints/deep_cfr_iter{:05}_model.json", iter);
            let s = format!("checkpoints/deep_cfr_iter{:05}_strategy.json", iter);
            let _ = trainer.save_models(&a, &s);
            println!("   💾 Đã lưu snapshot định kỳ: {}\n", a);
        }
    }

    println!("   ✅ Training complete in {:.1}s", train_start.elapsed().as_secs_f64());
    let _ = trainer.save_models(LAST_ADV, LAST_STRAT);

    println!("\n[2] 🏁 Final benchmark on BEST checkpoint...");
    let adv = MLP::load_from_file(BEST_ADV).unwrap_or_else(|_| trainer.advantage_net.read().unwrap().clone());
    let best_policy = Arc::new(DeepCFRPolicy::new(Arc::new(RwLock::new(adv)), false));

    if use_icm {
        println!("   🏆 Đang kiểm tra 500 giải đấu Tournament 8-max (50 BB khởi điểm, đánh đến khi hết tour)...");
        let dummy_champ = DeepCFRPolicy::new(Arc::new(RwLock::new(current_champion_net.clone())), false);
        let final_tour = evaluate_against_champion_tournament(
            best_policy.as_ref(),
            &dummy_champ,
            500,
            initial_bb,
            max_players,
        );
        println!(
            "   📊 Tổng kết 500 Tournaments: Vô địch: {} vs {} (Champ) | ITM: {} | Rank TB: {:.2} vs {:.2} | ICM Payout: {:+.3} | TB {:.1} ván/tour",
            final_tour.challenger_first_places,
            final_tour.champion_first_places,
            final_tour.challenger_itm_count,
            final_tour.challenger_avg_rank,
            final_tour.champion_avg_rank,
            final_tour.challenger_avg_icm_payout,
            final_tour.avg_hands_per_tour
        );
    }

    let final_res = run_suite(best_policy.as_ref(), 10_000, &depths);
    print_table("BEST checkpoint vs Baselines", &final_res);
    let _ = append_csv(BENCH_CSV, total_iterations, "final_best", &final_res);

    println!("\n[3] 🏆 LIVE SIMULATION: 1 GIẢI ĐẤU MẪU {}-MAX (50 BB) ĐÁNH ĐẾN KHI HẾT TOUR...", max_players);
    simulate_demo_tournament(best_policy.as_ref(), max_players, initial_bb, &mut rng);

    println!("\n[4] ⚡️ TOURNAMENT SUITE: MÔ PHỎNG 1,000 GIẢI ĐẤU ĐO HIỆU NĂNG TOÀN DIỆN...");
    let t_start = Instant::now();
    let num_sim_tours = 1000;

    let results: Vec<(usize, usize)> = (0..num_sim_tours)
        .into_par_iter()
        .map(|_| {
            let mut trng = thread_rng();
            simulate_hero_tournament(best_policy.as_ref(), max_players, initial_bb, &mut trng)
        })
        .collect();

    let mut finishes = [0usize; 8];
    let mut total_hands = 0usize;

    for &(rank, hands) in &results {
        if rank >= 1 && rank <= 8 {
            finishes[rank - 1] += 1;
        }
        total_hands += hands;
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    let itm = finishes[0] + finishes[1] + finishes[2];
    let avg_rank: f64 = finishes.iter().enumerate().map(|(i, &c)| (i + 1) as f64 * c as f64).sum::<f64>() / num_sim_tours as f64;

    println!("   🏆 Tổng số giải đấu hoàn tất: {} giải trong {:.2}s ({:.1} tours/sec)", num_sim_tours, elapsed, num_sim_tours as f64 / elapsed);
    println!("   🥇 Vô địch (Top 1)        : {:4} lần ({:5.1}%) [Kỳ vọng ngẫu nhiên: 12.5%]", finishes[0], finishes[0] as f64 * 100.0 / num_sim_tours as f64);
    println!("   🥈 Á quân  (Top 2)        : {:4} lần ({:5.1}%)", finishes[1], finishes[1] as f64 * 100.0 / num_sim_tours as f64);
    println!("   🥉 Hạng 3   (Top 3)        : {:4} lần ({:5.1}%)", finishes[2], finishes[2] as f64 * 100.0 / num_sim_tours as f64);
    println!("   💰 Tỷ lệ vào tiền (ITM Top 3): {:4} lần ({:5.1}%)", itm, itm as f64 * 100.0 / num_sim_tours as f64);
    println!("   💀 Bị loại sớm (Hạng 7-8)  : {:4} lần ({:5.1}%)", finishes[6] + finishes[7], (finishes[6] + finishes[7]) as f64 * 100.0 / num_sim_tours as f64);
    println!("   📊 Thứ hạng trung bình     : {:.2} / {}", avg_rank, max_players);
    println!("   ⏱️  Số ván trung bình/tour  : {:.1} ván", total_hands as f64 / num_sim_tours as f64);
    println!("\n📈 Benchmark history: {}", BENCH_CSV);
}

//! 🚀 DEEP CFR RESUME & CONTINUOUS TOURNAMENT TRAINER
//!
//! Chương trình chuyên dụng để Resume / Train tiếp Deep CFR từ checkpoint có sẵn:
//! - Tự động nhận diện checkpoint mới nhất và số iteration bắt đầu từ filename hoặc benchmark CSV.
//! - Kiến trúc mạng chuẩn [256, 256, 128] tương thích 100% với model hiện tại (176 features).
//! - Đồng bộ toàn diện cơ chế Multi-Depth Booster (50/25/12/8 BB) & 35% Heads-Up rèn luyện sát thủ 1-vs-1.
//! - Cơ chế Tournament Arena Gating (đánh full Sit & Go đến khi hết tour) để soán ngôi Champion chuẩn mực.
//! - Cập nhật trực tiếp vào deep_cfr_model.json khi Challenger vượt trội để bot chiến đấu ngay lập tức!

use poker_tournament::benchmark::{
    append_csv, evaluate_against_champion_table, evaluate_against_champion_tournament,
    print_table, run_suite, score, simulate_demo_tournament, simulate_hero_tournament,
    TournamentArenaResult,
};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::FEATURE_DIM;
use poker_tournament::neural_network::MLP;
use rand::thread_rng;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const HIDDEN_DIMS: [usize; 3] = [256, 256, 128];
const REPLAY_CAPACITY: usize = 2_500_000;

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_opt<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

/// Tự động dò tìm file strategy tương ứng từ file advantage model
fn guess_strategy_path(model_path: &str) -> String {
    if model_path.ends_with("_model.json") {
        model_path.replace("_model.json", "_strategy.json")
    } else if model_path.ends_with(".json") {
        model_path.replace(".json", "_strategy.json")
    } else {
        format!("{}_strategy.json", model_path)
    }
}

/// Tự động phát hiện số iteration xuất phát từ tên file checkpoint (ví dụ: iter01300 -> 1300)
fn detect_iter_from_filename(path: &str) -> Option<usize> {
    let lower = path.to_lowercase();
    if let Some(pos) = lower.find("iter") {
        let remainder = &lower[pos + 4..];
        let digits: String = remainder
            .chars()
            .skip_while(|c| *c == '_' || *c == '-')
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse::<usize>().ok()
    } else {
        None
    }
}

/// Tự động phát hiện iteration cao nhất đã ghi nhận trong file benchmark CSV
fn detect_iter_from_csv(path: &str) -> Option<usize> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut max_iter = None;
    for line in content.lines().rev() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("iter") {
            continue;
        }
        if let Some(first) = line.split(',').next() {
            if let Ok(it) = first.trim().parse::<usize>() {
                max_iter = Some(it);
                break;
            }
        }
    }
    max_iter
}

fn print_help() {
    println!("============================================================");
    println!("🚀 POKER AI DEEP CFR - RESUME / CONTINUE TRAINING TOOL 🚀");
    println!("============================================================");
    println!("Cú pháp: cargo run --release --bin resume -- [TÙY CHỌN]");
    println!();
    println!("Tùy chọn mô hình & Checkpoint:");
    println!("  --from <FILE>         File Advantage Net checkpoint để resume (mặc định: deep_cfr_last_model.json hoặc deep_cfr_model.json)");
    println!("  --from-strat <FILE>   File Strategy Net tương ứng (mặc định: tự suy đoán từ --from)");
    println!("  --champ <FILE>        File Champion gốc để đối đầu Arena (mặc định: nạp theo checkpoint ban đầu)");
    println!("  --out <NAME>          Tên tiền tố xuất mô hình (mặc định: deep_cfr)");
    println!("  --start-iter <N>      Iteration bắt đầu (tự động nhận diện từ tên file hoặc CSV nếu không chỉ định)");
    println!("  --iters <N>           Số iteration cần train tiếp (mặc định: 500)");
    println!();
    println!("Tùy chọn môi trường huấn luyện:");
    println!("  --players <N>         Quy mô bàn: 6 hoặc 8 (mặc định: 8)");
    println!("  --stack <N>           Stack BB khởi điểm mỗi người (mặc định: 50 BB)");
    println!("  --lr <F>              Tốc độ học Base Learning Rate (mặc định: 0.0005)");
    println!("  --traversals <N>      Số ván duyệt cây mỗi iter (mặc định: 250)");
    println!("  --train-steps <N>     Số bước SGD cập nhật mạng mỗi iter (mặc định: 400)");
    println!("  --batch-size <N>      Kích thước batch SGD (mặc định: 512)");
    println!("  --warmup <N>          Số iter khởi động gom mẫu vào buffer trước khi train mạng (mặc định: 5)");
    println!("  --chip-ev / --no-icm  Dùng Chip-EV thuần thay vì Tournament ICM (mặc định: BẬT ICM)");
    println!();
    println!("Tùy chọn Arena & Đánh giá:");
    println!("  --bench-every <N>     Tần suất kiểm tra Đấu trường Arena (mặc định: mỗi 10 iters)");
    println!("  --arena-tours <N>     Số giải Sit & Go full trong Arena Tournament Gating (mặc định: 100)");
    println!("  --bench-hands <N>     Số ván benchmark với các bot tĩnh (mặc định: 6000)");
    println!("  --snapshot-every <N>  Tần suất lưu checkpoint định kỳ vào checkpoints/ (mặc định: 100)");
    println!();
    println!("Ví dụ thực chiến:");
    println!("  1. Train tiếp tục tự động từ mô hình hiện tại thêm 500 iters:");
    println!("     cargo run --release --bin resume");
    println!();
    println!("  2. Resume từ một checkpoint cụ thể trong thư mục checkpoints/:");
    println!("     cargo run --release --bin resume -- --from checkpoints/deep_cfr_iter01300_model.json --iters 300");
    println!();
    println!("  3. Train thử nghiệm độc lập ra bộ file riêng (không đè model chính):");
    println!("     cargo run --release --bin resume -- --out test_experiment --iters 200");
    println!("============================================================");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // 1. Phân giải đường dẫn checkpoint nguồn
    let default_from = if std::path::Path::new("deep_cfr_last_model.json").exists() {
        "deep_cfr_last_model.json".to_string()
    } else {
        "deep_cfr_model.json".to_string()
    };
    let from_adv: String = arg(&args, "--from", default_from);

    let default_strat = guess_strategy_path(&from_adv);
    let from_strat: String = arg(&args, "--from-strat", default_strat);

    let champ_arg: Option<String> = arg_opt(&args, "--champ");
    let out: String = arg(&args, "--out", "deep_cfr".to_string());

    let max_players: usize = arg(&args, "--players", arg(&args, "--max-players", 8usize));
    let initial_bb: u32 = arg(&args, "--stack", 50u32);
    let iters: usize = arg(&args, "--iters", 500usize);
    let warmup: usize = arg(&args, "--warmup", 5usize);
    let lr: f32 = arg(&args, "--lr", 0.0005f32);
    let traversals: usize = arg(&args, "--traversals", 250usize);
    let train_steps: usize = arg(&args, "--train-steps", 400usize);
    let batch_size: usize = arg(&args, "--batch-size", 512usize);
    let bench_every: usize = arg(&args, "--bench-every", 10usize);
    let arena_tours: usize = arg(&args, "--arena-tours", 100usize);
    let arena_hands_cash: usize = arg(&args, "--arena-hands", 6000usize);
    let bench_hands: usize = arg(&args, "--bench-hands", 6000usize);
    let snapshot_every: usize = arg(&args, "--snapshot-every", 100usize);
    let use_icm: bool = !args.iter().any(|a| a == "--no-icm" || a == "--chip-ev");

    let depths = [50u32, 25u32, 12u32, 8u32];

    let csv_path = if out == "deep_cfr" {
        "benchmark_log.csv".to_string()
    } else {
        format!("{}_benchmark.csv", out)
    };

    // 2. Xác định iteration bắt đầu
    let start_iter: usize = match arg_opt(&args, "--start-iter") {
        Some(val) => val,
        None => {
            if let Some(file_it) = detect_iter_from_filename(&from_adv) {
                file_it
            } else if let Some(csv_it) = detect_iter_from_csv(&csv_path) {
                csv_it
            } else {
                0
            }
        }
    };

    let best_path = format!("{}_model.json", out);
    let best_strat_path = format!("{}_strategy.json", out);
    let last_adv = format!("{}_last_model.json", out);
    let last_strat = format!("{}_last_strategy.json", out);

    println!("============================================================");
    let mode_desc = if use_icm {
        "TOURNAMENT ICM (FULL RUN TO WINNER)"
    } else {
        "CHIP-EV THUẦN (MAX EV)"
    };
    println!("🔁 RESUME & CONTINUOUS DEEP CFR ({})", mode_desc);
    println!("⚙️  Bàn: {}-Max | Stack khởi điểm: {} BB | Chế độ: {}", max_players, initial_bb, mode_desc);
    println!("📂 Checkpoint nguồn: Advantage: {} | Strategy: {}", from_adv, from_strat);
    println!("📍 Điểm xuất phát: Iteration {:05} -> Train thêm {} iters (Tổng: {})", start_iter, iters, start_iter + iters);
    println!("💾 Tiền tố lưu: [{}] (CSV: {})", out, csv_path);
    println!("============================================================");

    // Chuẩn bị equity table preflop
    let _ = poker_tournament::equity::preflop_table();

    let mut rng = thread_rng();
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &HIDDEN_DIMS, REPLAY_CAPACITY, use_icm);
    trainer.max_players = max_players;
    trainer.fixed_players = None; // Cho phép giải đấu loại dần từ max_players về 2 người
    trainer.use_icm = use_icm;
    trainer.stack_bb = initial_bb; // Kích hoạt Tournament Multi-Depth Booster (50/25/12/8 BB + 35% Heads-Up)
    trainer.is_cash_game = false;
    trainer.lr = lr;
    trainer.iteration = start_iter;

    if !std::path::Path::new(&from_adv).exists() {
        eprintln!("❌ Lỗi: Không tìm thấy file checkpoint: {}", from_adv);
        std::process::exit(1);
    }

    match trainer.load_models(&from_adv, &from_strat) {
        Ok(has_strat) => println!(
            "   ✅ Nạp thành công model từ {} (Strategy net: {})",
            from_adv,
            if has_strat { "Có sẵn" } else { "Khởi tạo mới" }
        ),
        Err(e) => {
            eprintln!("❌ Lỗi nạp mô hình: {}", e);
            std::process::exit(1);
        }
    }

    // Thiết lập Champion cơ sở để làm thước đo soán ngôi
    let champ_file = champ_arg.unwrap_or_else(|| from_adv.clone());
    let current_champion_net: MLP = if std::path::Path::new(&champ_file).exists() && champ_file != from_adv {
        println!("   👑 Nạp Champion đối đầu độc lập từ: {}", champ_file);
        MLP::load_from_file(&champ_file).unwrap_or_else(|_| trainer.advantage_net.read().unwrap().clone())
    } else {
        println!("   👑 Dùng mô hình hiện tại làm mốc Champion xuất phát để Challenger soán ngôi.");
        trainer.advantage_net.read().unwrap().clone()
    };

    let total = start_iter + iters;
    let t_all = Instant::now();

    println!(
        "\n🚀 Bắt đầu quá trình huấn luyện: traversals={} | steps={} | batch={} | warmup={} iters",
        traversals, train_steps, batch_size, warmup
    );

    for k in 1..=iters {
        let global = start_iter + k;
        let warm = k <= warmup;
        let steps = if warm { 0 } else { train_steps };

        let t0 = Instant::now();
        let loss = trainer.step_iteration(traversals, steps, batch_size, &mut rng);
        let loss_str = if warm {
            format!("Warmup ({}/{} - Gom dữ liệu buffer)", k, warmup)
        } else {
            format!("Loss: {:.5}", loss)
        };

        println!(
            "   📍 Iter {:05}/{} | Adv: {:7} | Strat: {:7} | {} | {:.1}s",
            global, total,
            trainer.adv_memory.len(), trainer.strategy_memory.len(),
            loss_str, t0.elapsed().as_secs_f64()
        );

        if !warm && (k % bench_every == 0 || global % bench_every == 0) {
            let bt = Instant::now();
            let challenger_policy = DeepCFRPolicy::new(Arc::clone(&trainer.advantage_net), false);
            let champion_policy = DeepCFRPolicy::new(Arc::new(RwLock::new(current_champion_net.clone())), false);

            if use_icm {
                println!(
                    "\n   ⚔️  ĐẤU TRƯỜNG TOURNAMENT {}-MAX ({} BB Khởi Điểm - Đánh Đến Khi Hết Tour): Challenger (Iter {}) vs Champion...",
                    max_players, initial_bb, global
                );
                let tour_res: TournamentArenaResult = evaluate_against_champion_tournament(
                    &challenger_policy,
                    &champion_policy,
                    arena_tours,
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
                        tour_res.challenger_avg_rank, tour_res.champion_avg_rank, best_path
                    );
                    let _ = trainer.save_models(&best_path, &best_strat_path);
                    if out != "deep_cfr" {
                        println!("   🌟 Đã lưu Champion vào {} & {}", best_path, best_strat_path);
                    }
                } else {
                    println!("   🛡️  Champion giữ vững ngôi vương. Challenger chưa đủ vượt trội (hoặc TrashShove > 5%).");
                }
            } else {
                println!(
                    "\n   ⚔️  ĐẤU TRƯỜNG {}-MAX ({} BB): Challenger (Iter {}) vs Champion...",
                    max_players, initial_bb, global
                );
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
                        arena_res.challenger_bb_per_100, best_path
                    );
                    let _ = trainer.save_models(&best_path, &best_strat_path);
                } else {
                    println!("   🛡️  Champion giữ vững ngôi vương. Challenger chưa đủ vượt trội (hoặc TrashShove > 5%).");
                }
            }

            let r_adv = run_suite(&challenger_policy, bench_hands, &depths);
            print_table("Tham khảo đối thủ tĩnh", &r_adv);
            let _ = append_csv(&csv_path, global, "adv", &r_adv);

            let s = score(&r_adv);
            println!("   🧪 Benchmark log score: {:+.1} bb/100\n", s);

            // Cập nhật snapshot mới nhất
            let _ = trainer.save_models(&last_adv, &last_strat);
        }

        // Lưu checkpoint định kỳ vào thư mục checkpoints/
        if global % snapshot_every == 0 {
            let _ = std::fs::create_dir_all("checkpoints");
            let a = format!("checkpoints/{}_iter{:05}_model.json", out, global);
            let s = format!("checkpoints/{}_iter{:05}_strategy.json", out, global);
            let _ = trainer.save_models(&a, &s);
            println!("   💾 Đã lưu snapshot định kỳ: {}\n", a);
        }
    }

    println!("\n============================================================");
    println!("🎉 QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT TRONG {:.1}s!", t_all.elapsed().as_secs_f64());
    println!("💾 Checkpoint mới nhất: {} & {}", last_adv, last_strat);
    println!("============================================================");

    let _ = trainer.save_models(&last_adv, &last_strat);

    // Đánh giá tổng kết mô hình Champion tốt nhất
    println!("\n[1] 🏁 Final benchmark trên BEST CHAMPION...");
    let best_adv_to_load = if std::path::Path::new(&best_path).exists() {
        best_path.clone()
    } else {
        last_adv.clone()
    };

    let adv = MLP::load_from_file(&best_adv_to_load)
        .unwrap_or_else(|_| trainer.advantage_net.read().unwrap().clone());
    let best_policy = Arc::new(DeepCFRPolicy::new(Arc::new(RwLock::new(adv)), false));

    if use_icm {
        println!("   🏆 Đang kiểm tra 500 giải đấu Tournament {}-max (50 BB khởi điểm)...", max_players);
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
    print_table("BEST Checkpoint vs Baselines", &final_res);
    let _ = append_csv(&csv_path, total, "final_best", &final_res);

    println!("\n[2] 🏆 LIVE SIMULATION: 1 GIẢI ĐẤU MẪU {}-MAX ({} BB)...", max_players, initial_bb);
    simulate_demo_tournament(best_policy.as_ref(), max_players, initial_bb, &mut rng);

    println!("\n[3] ⚡️ TOURNAMENT SUITE: MÔ PHỎNG 1,000 GIẢI ĐẤU ĐO HIỆU NĂNG TOÀN DIỆN...");
    let t_sim = Instant::now();
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

    let elapsed = t_sim.elapsed().as_secs_f64();
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
    println!("\n📈 Lịch sử benchmark đã lưu tại: {}", csv_path);
}

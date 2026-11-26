//! 🚀 DEEP CFR LOUNGE CASH GAME TRAINER (REAL CHIP / CHIP-EV / 12-300+ BB)
//!
//! Chương trình huấn luyện chuyên dụng cho chế độ Lounge PokerNow / Real Cash Game:
//! - Bàn chơi tiền thật (Lounge / Ring Game): Không có cơ chế giải đấu Tournament hay loại dần.
//! - Hàm thưởng thuần Chip-EV (không dùng Tournament ICM, mỗi 1 chip = tiền thật).
//! - Phân phối Stack cực kỳ phong phú và biến thiên ngẫu nhiên giữa các người chơi:
//!   • Min Buy-in: 12 BB (Short-stackers).
//!   • Standard Buy-in: 40 - 100 BB.
//!   • Deep-stack tích lũy: 100 - 300+ BB (hỗ trợ dải mượt tới 300+ BB).
//! - Hỗ trợ quy mô bàn linh hoạt từ 2 đến 10 người (tổng quát từ Heads-Up đến 10-Max).
//! - Kiến trúc mạng nơ-ron mặc định [256, 256, 256] với 187 features chuẩn 10-Max.
//! - Độc lập hoàn toàn: Lưu ra file `deep_cfr_lounge_model.json` và `deep_cfr_lounge_strategy.json`.

use poker_tournament::benchmark::{
    append_csv, evaluate_against_champion_cash, print_table, run_cash_suite, ArenaMatchResult,
};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::{FEATURE_DIM, MAX_PLAYERS};
use poker_tournament::neural_network::MLP;
use rand::thread_rng;
use std::io::Write;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const BEST_ADV: &str = "deep_cfr_lounge_model.json";
const BEST_STRAT: &str = "deep_cfr_lounge_strategy.json";
const LAST_ADV: &str = "deep_cfr_lounge_last_model.json";
const LAST_STRAT: &str = "deep_cfr_lounge_last_strategy.json";
const BENCH_CSV: &str = "benchmark_lounge_log.csv";
const REPLAY_CAPACITY: usize = 3_000_000;

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

fn guess_strategy_path(model_path: &str) -> String {
    if model_path.ends_with("_model.json") {
        model_path.replace("_model.json", "_strategy.json")
    } else if model_path.ends_with(".json") {
        model_path.replace(".json", "_strategy.json")
    } else {
        format!("{}_strategy.json", model_path)
    }
}

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

fn detect_iter_from_csv(path: &str) -> Option<usize> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines().rev() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("iter") {
            continue;
        }
        if let Some(first) = line.split(',').next() {
            if let Ok(it) = first.trim().parse::<usize>() {
                return Some(it);
            }
        }
    }
    None
}

fn print_help() {
    println!("============================================================");
    println!("🚀 POKER AI DEEP CFR - LOUNGE CASH GAME TRAINER (2-10 MAX) 🚀");
    println!("============================================================");
    println!("Cú pháp: cargo run --release --bin train_lounge -- [TÙY CHỌN]");
    println!();
    println!("Tùy chọn huấn luyện:");
    println!("  --players <N>         Quy mô bàn tối đa: 2 đến 10 người (mặc định: 10)");
    println!("  --fixed-players       Khóa cứng số người mỗi ván đúng bằng --players (mặc định: TẮT, phủ 2..N)");
    println!("  --iters <N>           Số iteration cần chạy (mặc định: 10,000)");
    println!("  --traversals <N>      Số ván duyệt cây mỗi iter (mặc định: 250)");
    println!("  --train-steps <N>     Số bước SGD train mạng mỗi iter (mặc định: 400)");
    println!("  --batch-size <N>      Batch size SGD (mặc định: 512)");
    println!("  --lr <F>              Learning rate cơ sở (mặc định: 0.0005)");
    println!("  --hidden <N>          Kích thước mỗi tầng ẩn khi --fresh (mặc định: 256)");
    println!("  --warmup <N>          Số iter gom mẫu trước khi SGD (mặc định: 5 khi resume, 0 khi --fresh)");
    println!("  --start-iter <N>      Iteration bắt đầu (tự nhận diện từ tên file/CSV nếu resume)");
    println!("  --sparring            Bật đấu tập với 8 trường phái Quân Xanh (mặc định: TẮT)");
    println!("  --fresh               Bắt đầu train mạng mới hoàn toàn (bỏ qua model cũ)");
    println!();
    println!("Tùy chọn Đấu trường Arena & Benchmark:");
    println!("  --bench-every <N>     Tần suất kiểm tra Arena & Benchmark (mặc định: mỗi 10 iters)");
    println!("  --arena-hands <N>     Số ván đấu trường Duplicate Cash đối đầu Champion (mặc định: 6,000)");
    println!("  --bench-hands <N>     Số ván benchmark đối thủ tĩnh mỗi mức (mặc định: 2,000)");
    println!("  --snapshot-every <N>  Tần suất lưu snapshot vào checkpoints/ (mặc định: 100)");
    println!();
    println!("Tùy chọn Checkpoint:");
    println!("  --from <FILE>         File Advantage Net checkpoint để resume");
    println!("  --from-strat <FILE>   File Strategy Net tương ứng");
    println!("============================================================");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    let max_players: usize = arg(&args, "--players", arg(&args, "--max-players", 10usize)).clamp(2, MAX_PLAYERS);
    let fixed_players_flag = args.iter().any(|a| a == "--fixed-players");
    let iters: usize = arg(&args, "--iters", 10_000usize);
    let traversals: usize = arg(&args, "--traversals", 250usize);
    let train_steps: usize = arg(&args, "--train-steps", 400usize);
    let batch_size: usize = arg(&args, "--batch-size", 512usize);
    let bench_every: usize = arg(&args, "--bench-every", 10usize);
    let arena_hands: usize = arg(&args, "--arena-hands", 6000usize);
    let bench_hands: usize = arg(&args, "--bench-hands", 2000usize);
    let snapshot_every: usize = arg(&args, "--snapshot-every", 100usize);
    let lr: f32 = arg(&args, "--lr", 0.0005f32);
    let hidden_size: usize = arg(&args, "--hidden", 256usize);
    let enable_sparring = args.iter().any(|a| a == "--sparring" || a == "--spar" || a == "--with-sparring");
    let fresh = args.iter().any(|a| a == "--fresh" || a == "--new" || a == "--from-scratch");

    let default_from = if std::path::Path::new(LAST_ADV).exists() {
        LAST_ADV.to_string()
    } else {
        BEST_ADV.to_string()
    };
    let from_adv: String = arg(&args, "--from", default_from);
    let default_strat = guess_strategy_path(&from_adv);
    let from_strat: String = arg(&args, "--from-strat", default_strat);

    let can_resume = !fresh && std::path::Path::new(&from_adv).exists();
    let default_warmup = if can_resume { 5usize } else { 0usize };
    let warmup: usize = arg(&args, "--warmup", default_warmup);

    let mut start_iter: usize = if fresh {
        0
    } else if let Some(val) = arg_opt(&args, "--start-iter") {
        val
    } else if let Some(file_it) = detect_iter_from_filename(&from_adv) {
        file_it
    } else if let Some(csv_it) = detect_iter_from_csv(BENCH_CSV) {
        csv_it
    } else {
        0
    };

    let hidden_dims: Vec<usize> = if can_resume {
        if let Ok(m) = MLP::load_from_file(&from_adv) {
            let dims = m.layer_dims();
            if dims.len() >= 3 && dims[0] == FEATURE_DIM {
                dims[1..dims.len() - 1].to_vec()
            } else {
                vec![hidden_size, hidden_size, hidden_size]
            }
        } else {
            vec![hidden_size, hidden_size, hidden_size]
        }
    } else {
        vec![hidden_size, hidden_size, hidden_size]
    };

    println!("============================================================");
    println!("🔥 DEEP CFR LOUNGE CASH GAME TRAINER (REAL CHIP / CHIP-EV) 🔥");
    println!("============================================================");
    println!(
        "  • Quy mô bàn       : Bàn {}-Max Lounge ({})",
        max_players,
        if fixed_players_flag {
            format!("Khóa cứng {} người", max_players)
        } else {
            format!("Linh hoạt 2..{} người", max_players)
        }
    );
    println!("  • Cơ chế thưởng    : Thuần Chip-EV (Mỗi chip = Tiền thật, không ICM)");
    println!("  • Phân bổ Stack    : Random ngẫu nhiên từng ghế (Min 12 BB, Standard 40-100 BB, Deep 100-300+ BB)");
    println!("  • Feature Scaling  : Cơ số chuẩn 150 BB (Dải mượt 0..300+ BB, {} features)", FEATURE_DIM);
    println!("  • Cấu trúc Mạng    : {:?} -> {}", hidden_dims, poker_tournament::environment::NUM_ACTIONS);
    println!("  • Quân Xanh        : {}", if enable_sparring { "BẬT (50% Đấu Tập Quân Xanh / 50% Self-Play)" } else { "TẮT (100% Thuần Self-Play)" });
    println!("  • Warmup / Start   : Warmup {} iters | Start Iter {} -> {}", warmup, start_iter, start_iter + iters);
    println!("  • File Output      : {} & {}", BEST_ADV, BEST_STRAT);
    println!("============================================================");

    println!("\n[0] 📐 Preparing preflop equity table (169 classes x 1..=9 opps)...");
    let t0 = Instant::now();
    let _ = poker_tournament::equity::preflop_table();
    println!("   ✅ Ready in {:.1}s\n", t0.elapsed().as_secs_f64());

    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, REPLAY_CAPACITY, false);
    trainer.is_cash_game = true;
    trainer.max_players = max_players;
    trainer.fixed_players = if fixed_players_flag { Some(max_players) } else { None };
    trainer.use_icm = false;
    trainer.stack_bb = 0; // Kích hoạt nhánh Cash/Lounge Stack Random (Min 12 BB -> Deep 300+ BB)
    trainer.enable_sparring = enable_sparring;
    trainer.lr = lr;

    let mut current_champion_strat: Option<MLP> = None;

    if can_resume {
        println!("📥 Đang nạp checkpoint có sẵn: {}", from_adv);
        match trainer.load_models(&from_adv, &from_strat) {
            Ok(true) => {
                println!("   ✅ Nạp thành công Advantage Net và Strategy Net ({})!", from_strat);
                trainer.iteration = start_iter;
                current_champion_strat = trainer.strategy_net.read().ok().map(|n| (*n).clone());
            }
            Ok(false) => {
                println!("   ✅ Nạp Advantage Net (Strategy Net khởi tạo mới)!");
                trainer.iteration = start_iter;
            }
            Err(e) => {
                println!("   ⚠️  Không thể nạp checkpoint ({}), bắt đầu train mạng mới hoàn toàn.", e);
                start_iter = 0;
                trainer.iteration = 0;
                let _ = std::fs::remove_file(BEST_STRAT);
                let _ = std::fs::remove_file(LAST_STRAT);
                let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
            }
        }
    } else {
        println!("🌱 Khởi tạo mạng mới hoàn toàn cho Lounge Cash Game (Zero-bias).");
        start_iter = 0;
        trainer.iteration = 0;
        let _ = std::fs::remove_file(BEST_STRAT);
        let _ = std::fs::remove_file(LAST_STRAT);
        let _ = std::fs::remove_file(BENCH_CSV);
        let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
    }

    // Nếu resume từ LAST_ADV nhưng BEST_ADV khác tồn tại & hợp lệ, ưu tiên dùng BEST làm mốc Champion
    let (mut current_champion_net, loaded_best_strat) = if can_resume
        && from_adv != BEST_ADV
        && std::path::Path::new(BEST_ADV).exists()
    {
        if let Ok(best_mlp) = MLP::load_from_file(BEST_ADV) {
            if best_mlp.layer_dims() == trainer.advantage_net.read().unwrap().layer_dims() {
                let s = MLP::load_from_file(BEST_STRAT)
                    .ok()
                    .filter(|sm| sm.layer_dims() == best_mlp.layer_dims());
                println!("   👑 Đã nạp mốc Champion độc lập từ: {}", BEST_ADV);
                (best_mlp, s)
            } else {
                (trainer.advantage_net.read().unwrap().clone(), current_champion_strat.clone())
            }
        } else {
            (trainer.advantage_net.read().unwrap().clone(), current_champion_strat.clone())
        }
    } else {
        (trainer.advantage_net.read().unwrap().clone(), current_champion_strat.clone())
    };
    current_champion_strat = loaded_best_strat;

    let mut rng = thread_rng();
    let train_start = Instant::now();
    let total = start_iter + iters;

    for k in 1..=iters {
        let global = start_iter + k;
        let warm = k <= warmup;
        let steps = if warm { 0 } else { train_steps };

        let it0 = Instant::now();
        let adv_loss = trainer.step_iteration(traversals, steps, batch_size, &mut rng);
        let loss_str = if warm {
            format!("Warmup ({}/{})", k, warmup)
        } else {
            format!("Loss: {:.5}", adv_loss)
        };
        println!(
            "   📍 Lounge Iter {:05}/{} | Adv: {:7} | Strat: {:7} | {} | {:.1}s",
            global,
            total,
            trainer.adv_memory.len(),
            trainer.strategy_memory.len(),
            loss_str,
            it0.elapsed().as_secs_f64()
        );

        if !warm && bench_every > 0 && (global % bench_every == 0 || k == iters) {
            let bt = Instant::now();
            println!(
                "\n   ⚔️  ĐẤU TRƯỜNG LOUNGE {}-MAX (Duplicate Deals, Chip-EV, 12-300+ BB): Challenger (Iter {}) vs Champion...",
                max_players, global
            );
            let challenger_policy = trainer.get_policy(false);
            let champion_policy = DeepCFRPolicy::with_strategy(
                Arc::new(RwLock::new(current_champion_net.clone())),
                current_champion_strat.clone(),
                false,
            );

            let arena_res: ArenaMatchResult = evaluate_against_champion_cash(
                &challenger_policy,
                &champion_policy,
                arena_hands,
                max_players,
            );

            println!(
                "   📊 Kết quả Arena Lounge (Duplicate): {:+.2} ± {:.2} bb/100 | TrashShove: {:.2}% | {} ván trong {:.1}s",
                arena_res.challenger_bb_per_100,
                arena_res.ci95,
                arena_res.challenger_trash_shove_pct,
                arena_res.hands,
                bt.elapsed().as_secs_f64()
            );

            if arena_res.is_winner {
                println!(
                    "   👑 SOÁN NGÔI CHAMPION! Iter {} thắng vượt trội {:+.2} bb/100 -> Đã cập nhật {}!",
                    global, arena_res.challenger_bb_per_100, BEST_ADV
                );
                current_champion_net = trainer.advantage_net.read().unwrap().clone();
                current_champion_strat = if trainer.strategy_trained {
                    trainer.strategy_net.read().ok().map(|n| (*n).clone())
                } else {
                    None
                };
                let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
            } else {
                println!("   🛡️  Champion vẫn giữ vững vị thế.");
            }

            if bench_hands > 0 {
                let r_cash = run_cash_suite(&challenger_policy, bench_hands, max_players);
                print_table("Lounge Cash vs Baselines", &r_cash);
                let _ = append_csv(BENCH_CSV, global, "adv", &r_cash);
            }

            let need_header = !std::path::Path::new(BENCH_CSV).exists();
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(BENCH_CSV) {
                if need_header {
                    let _ = writeln!(f, "iter,policy,opponent,stack_bb,hands,bb_per_100,ci95,fold_pct,shove_pct,trash_shove_pct");
                }
                let _ = writeln!(
                    f,
                    "{},lounge_arena,Champion,150,{},{:.2},{:.2},0.00,0.00,{:.2}",
                    global,
                    arena_res.hands,
                    arena_res.challenger_bb_per_100,
                    arena_res.ci95,
                    arena_res.challenger_trash_shove_pct
                );
            }

            // Luôn lưu checkpoint gần nhất
            let _ = trainer.save_models(LAST_ADV, LAST_STRAT);
            println!();
        }

        if snapshot_every > 0 && global % snapshot_every == 0 {
            let _ = std::fs::create_dir_all("checkpoints");
            let a = format!("checkpoints/lounge_iter{:05}_model.json", global);
            let s = format!("checkpoints/lounge_iter{:05}_strategy.json", global);
            let _ = trainer.save_models(&a, &s);
            println!("   💾 Đã lưu snapshot định kỳ: {}\n", a);
        }
    }

    let _ = trainer.save_models(LAST_ADV, LAST_STRAT);
    println!("🎉 Hoàn thành huấn luyện Lounge Cash Game trong {:.1} phút!", train_start.elapsed().as_secs_f64() / 60.0);
}

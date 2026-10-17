use poker_tournament::benchmark::{
    append_csv, evaluate_against_champion_table, print_table, run_suite, score,
};
use poker_tournament::deep_cfr::DeepCFRPolicy;
use poker_tournament::deep_cfr::DeepCFRTrainer;
use poker_tournament::environment::{Environment, FEATURE_DIM};
use poker_tournament::gui_bridge::TableViewModel;
use poker_tournament::neural_network::MLP;
use poker_tournament::self_play::SelfPlayEngine;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const BEST_ADV: &str = "deep_cfr_model.json";
const BEST_STRAT: &str = "deep_cfr_strategy.json";
const LAST_ADV: &str = "deep_cfr_last_model.json";
const LAST_STRAT: &str = "deep_cfr_last_strategy.json";
const BENCH_CSV: &str = "benchmark_log.csv";

/// Phân tích tham số dòng lệnh linh hoạt:
/// - Tham số 1: Số người chơi trên bàn (6 hoặc 8 người)
/// - Tham số 2: Hàm thưởng (ICM hoặc EV thuần)
fn parse_args() -> (usize, bool) {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("============================================================");
        println!("🚀 POKER AI DEEP CFR TRAINING - HƯỚNG DẪN CẤU HÌNH 🚀");
        println!("============================================================");
        println!("Cú pháp: cargo run --release -- [SỐ_NGƯỜI: 6|8] [CHẾ_ĐỘ: icm|ev]");
        println!("Hoặc:   cargo run --release -- [--players 6|8] [--icm|--ev]");
        println!();
        println!("Tham số 1: Quy mô bàn");
        println!("  6 hoặc --players 6 hoặc --6max : Bàn 6-Max (tối đa 6 người chơi)");
        println!("  8 hoặc --players 8 hoặc --8max : Bàn 8-Max (tối đa 8 người chơi)");
        println!();
        println!("Tham số 2: Hàm thưởng (Reward)");
        println!("  icm hoặc --icm                 : Bật Tournament ICM (Sit & Go, Payout Zero-Sum)");
        println!("  ev hoặc --ev hoặc --chip-ev    : Dùng Chip-EV thuần (Cash Game / EV Max)");
        println!();
        println!("Độ sâu stack:");
        println!("  🟢 Mỗi người chơi cầm 100 BB khi train (thay vì 50 BB max trước đây).");
        println!();
        println!("Ví dụ thực chiến:");
        println!("  1. Bàn 6-max, dùng EV thuần:");
        println!("     cargo run --release -- 6 ev");
        println!("  2. Bàn 6-max, dùng Tournament ICM:");
        println!("     cargo run --release -- 6 icm");
        println!("  3. Bàn 8-max, dùng Tournament ICM:");
        println!("     cargo run --release -- 8 icm");
        println!("  4. Bàn 8-max, dùng EV thuần:");
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
    let mode_desc = if use_icm { "TOURNAMENT ICM" } else { "CHIP-EV THUẦN" };

    println!("============================================================");
    println!("🚀 POKER AI DEEP CFR ({}-MAX ARENA GATING / 100 BB STACK) 🚀", max_players);
    println!("⚙️  CẤU HÌNH TRAIN: Bàn {}-Max | Chế độ: {} | Stack: 100 BB/người", max_players, mode_desc);
    println!("============================================================");

    let mut rng = rand::thread_rng();

    println!("\n[0] 📐 Preparing preflop equity table...");
    let t0 = Instant::now();
    let _ = poker_tournament::equity::preflop_table();
    println!("   ✅ Ready in {:.1}s", t0.elapsed().as_secs_f64());

    println!("\n[1] 🏋️ Deep CFR training with {}-Max Arena Gating & {} (100 BB Stack)...", max_players, mode_desc);
    
    // Kiến trúc mạng nơ-ron: 3 tầng ẩn đủ sâu cho 176 features
    let hidden_dims = [256, 256, 128];
    
    // Capacity 2.5M mẫu là vừa vặn bộ nhớ RAM
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, 2_500_000, use_icm);
    trainer.max_players = max_players; // 🟢 Bàn max 6 hoặc max 8 người
    trainer.fixed_players = None;      // Ưu tiên bàn max_players đã chọn (có cọ xát heads-up/short-handed)
    trainer.use_icm = use_icm;         // 🟢 Bật Tournament ICM hoặc dùng Chip-EV thuần
    trainer.stack_bb = 100;            // 🟢 Mỗi người chơi cầm 100 BB khi train thay vì 50 BB max
    trainer.lr = 0.0005;               // Giảm nhẹ learning rate để tránh dao động All-in rác

    let total_iterations = 10_000;     // Chạy liên tục 10,000 iterations
    let traversals_per_iter = 250;     // Duyệt cây mỗi iter
    let train_steps_per_iter = 400;    // Ép mạng học kỹ các tình huống
    let batch_size = 512;              // Kích thước vàng cho SIMD/Rayon

    let bench_every = 10;              // Benchmark và đấu trường Arena mỗi 10 iters
    let arena_hands = 6000;            // Đủ tin cậy để soán ngôi Champion
    let arena_stack_bb = 100;          // 🟢 Đấu trường Arena 100 BB thực chiến chuẩn
    let bench_hands = 6000;            // Đánh giá đối thủ tĩnh
    let depths = [100u32, 50u32, 25u32, 12u32]; // 🟢 Đầy đủ các mốc stack 100 BB, 50 BB, 25 BB và Push/Fold 12 BB!

    let _ = std::fs::remove_file(BENCH_CSV);

    let mut current_champion_net: MLP = trainer.advantage_net.read().unwrap().clone();
    let _ = trainer.save_models(BEST_ADV, BEST_STRAT);

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

            println!("\n   ⚔️  ĐẤU TRƯỜNG {}-MAX (100 BB): Challenger (Iter {}) vs Champion...", max_players, iter);
            let challenger_policy = DeepCFRPolicy::new(Arc::clone(&trainer.advantage_net), false);
            let champion_policy = DeepCFRPolicy::new(Arc::new(RwLock::new(current_champion_net.clone())), false);

            let arena_res = evaluate_against_champion_table(
                &challenger_policy,
                &champion_policy,
                arena_hands,
                arena_stack_bb,
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

            let r_adv = run_suite(&challenger_policy, bench_hands, &depths);
            print_table("Tham khảo đối thủ tĩnh", &r_adv);
            let _ = append_csv(BENCH_CSV, iter, "adv", &r_adv);

            let s = score(&r_adv);
            println!("   🧪 Benchmark log score: {:+.1} bb/100\n", s);

            // Tự động lưu checkpoint mới nhất phòng khi dừng máy giữa chừng
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

    let final_res = run_suite(best_policy.as_ref(), 10_000, &depths);
    print_table("BEST checkpoint (10k hands/cell)", &final_res);
    let _ = append_csv(BENCH_CSV, total_iterations, "final_best", &final_res);

    println!("\n[3] 🎯 Action probabilities on a {}-player hand (100 BB stack)...", max_players);
    let num_players = max_players;
    let (bb, sb, button_idx) = (100, 50, 0);
    let stacks = vec![100 * bb; num_players];
    let mut env = Environment::new(num_players, &stacks, bb, sb, button_idx);
    env.reset_hand(&mut rng, button_idx);
    let mask = env.get_action_mask();
    let table_vm = TableViewModel::from_game_state(&env.state, &mask, Some(best_policy.as_ref()), true);
    println!("{}", serde_json::to_string_pretty(&table_vm).unwrap());

    println!("\n[4] ⚡️ Self-play throughput...");
    let engine = SelfPlayEngine::new();
    let num_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(4);
    let sim_start = Instant::now();
    let rx = engine.run_parallel_simulation(Arc::clone(&best_policy), num_workers, 20_000, 10_000);
    let mut total_transitions = 0u64;
    while let Ok(_t) = rx.recv() {
        total_transitions += 1;
    }
    let elapsed = sim_start.elapsed().as_secs_f64();
    let simulated = engine.total_simulated_hands.load(std::sync::atomic::Ordering::Relaxed);
    println!("   🃏 {} hands, {} transitions, {:.1} hands/sec", simulated, total_transitions, simulated as f64 / elapsed);
    println!("\n📈 Benchmark history: {}", BENCH_CSV);
}

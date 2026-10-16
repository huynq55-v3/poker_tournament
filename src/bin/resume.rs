//! Train tiếp từ checkpoint với Đấu trường 8-Max Arena Gating & Tournament ICM
//!
//! cargo run --release --bin resume -- [--from F] [--from-strat F] [--start-iter N]
//!     [--iters N] [--out PREFIX] [--warmup N] [--lr X] [--arena-hands N]

use poker_tournament::benchmark::{
    append_csv, evaluate_against_champion, print_table, run_suite, score, BenchResult,
};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::FEATURE_DIM;
use poker_tournament::neural_network::MLP;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const HIDDEN: [usize; 3] = [256, 256, 128];
const DEPTHS: [u32; 3] = [50, 25, 12];

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn bench_policy(net: &Arc<RwLock<MLP>>, strat_net: Option<MLP>, hands: usize) -> Vec<BenchResult> {
    let policy = DeepCFRPolicy::with_strategy(Arc::clone(net), strat_net, false);
    run_suite(&policy, hands, &DEPTHS)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("============================================================");
        println!("📖 HƯỚNG DẪN SỬ DỤNG: RESUME DEEP CFR (8-MAX ARENA GATING)");
        println!("============================================================");
        println!("Cú pháp: cargo run --release --bin resume -- [TÙY CHỌN]\n");
        println!("Các tùy chọn:");
        println!("  --from <FILE>         File Advantage Net nguồn (mặc định: deep_cfr_model.json)");
        println!("  --from-strat <FILE>   File Strategy Net nguồn (mặc định: deep_cfr_strategy.json)");
        println!("  --out <TIỀN_TỐ>       Tiền tố lưu file output (mặc định: tour_cont)");
        println!("  --start-iter <N>      Iteration bắt đầu đánh số (mặc định: 160)");
        println!("  --iters <N>           Số iteration train thêm (mặc định: 300)");
        println!("  --warmup <N>          Số iter warmup chỉ thu thập mẫu (mặc định: 5)");
        println!("  --lr <F>              Learning rate (mặc định: 0.0003)");
        println!("  --traversals <N>      Số ván duyệt cây mỗi iter (mặc định: 250)");
        println!("  --train-steps <N>     Số bước SGD train mạng mỗi iter (mặc định: 350)");
        println!("  --arena-hands <N>     Số ván đấu trường Arena soán ngôi (mặc định: 6000)");
        println!("  --bench-hands <N>     Số ván đánh thử benchmark đối thủ tĩnh (mặc định: 4000)");
        println!("  --chip-ev / --no-icm  Tắt Tournament ICM, chuyển sang Chip-EV thuần (mặc định: BẬT ICM)");
        println!();
        println!("Ví dụ thực chiến:");
        println!("  1. Train tiếp chuẩn Tournament ICM từ iter 160 thêm 300 iter:");
        println!("     cargo run --release --bin resume -- --from deep_cfr_model.json --from-strat deep_cfr_strategy.json --start-iter 160 --iters 300 --out tour_cont");
        println!();
        println!("  2. Train tiếp và ghi đè trực tiếp Champion chính khi thắng Arena:");
        println!("     cargo run --release --bin resume -- --out deep_cfr --start-iter 160 --iters 300");
        println!("============================================================");
        return;
    }

    let from_adv: String = arg(&args, "--from", "deep_cfr_model.json".to_string());
    let from_strat: String = arg(&args, "--from-strat", "deep_cfr_strategy.json".to_string());
    let out: String = arg(&args, "--out", "tour_cont".to_string());
    let start_iter: usize = arg(&args, "--start-iter", 160);
    let iters: usize = arg(&args, "--iters", 300);
    let warmup: usize = arg(&args, "--warmup", 5);
    let lr: f32 = arg(&args, "--lr", 0.0003);
    let traversals: usize = arg(&args, "--traversals", 250);
    let train_steps: usize = arg(&args, "--train-steps", 350);
    let arena_hands: usize = arg(&args, "--arena-hands", 6000);
    let arena_stack_bb: u32 = arg(&args, "--arena-stack", 35);
    let bench_hands: usize = arg(&args, "--bench-hands", 4000);
    let use_icm: bool = !args.iter().any(|a| a == "--no-icm" || a == "--chip-ev");

    let batch = 512;
    let bench_every = 10;
    let snapshot_every = 50;

    let csv = format!("{}_benchmark.csv", out);
    let best_path = format!("{}_best_model.json", out);
    let best_strat_path = format!("{}_best_strategy.json", out);
    let last_adv = format!("{}_last_model.json", out);
    let last_strat = format!("{}_last_strategy.json", out);

    println!("============================================================");
    let mode_desc = if use_icm { "TOURNAMENT ICM & 8-MAX ARENA GATING" } else { "CHIP-EV & 8-MAX ARENA GATING" };
    println!("🔁 RESUME DEEP CFR ({})", mode_desc);
    println!("============================================================");

    let _ = poker_tournament::equity::preflop_table();

    let mut rng = rand::thread_rng();
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &HIDDEN, 1_500_000, use_icm);
    trainer.fixed_players = None;
    trainer.use_icm = use_icm;
    trainer.lr = lr;

    match trainer.load_models(&from_adv, &from_strat) {
        Ok(has_strat) => println!(
            "   ✅ Loaded {} (strategy net: {})",
            from_adv,
            if has_strat { "yes" } else { "no" }
        ),
        Err(e) => {
            eprintln!("   ❌ Không nạp được model: {}", e);
            std::process::exit(1);
        }
    }
    trainer.iteration = start_iter;

    // Giữ bản Champion gốc làm mốc đối đầu Arena
    let mut current_champion_net: MLP = trainer.advantage_net.read().unwrap().clone();
    let _ = trainer.save_models(&best_path, &best_strat_path);

    println!(
        "   ⚙️  start_iter={} iters={} warmup={} lr={} reward={} traversals={} steps={}",
        start_iter, iters, warmup, lr, if use_icm { "ICM" } else { "Chip-EV" }, traversals, train_steps
    );

    // Mốc ban đầu sau khi load model
    println!("\n[1] 📐 Đánh giá mốc ban đầu của Champion...");
    let base = bench_policy(&trainer.advantage_net, None, bench_hands);
    print_table("MỐC BAN ĐẦU (Champion Gốc)", &base);
    let _ = append_csv(&csv, start_iter, "champion_base", &base);

    let total = start_iter + iters;
    let t_all = Instant::now();

    for k in 1..=iters {
        let global = start_iter + k;
        let warm = k <= warmup;
        let steps = if warm { 0 } else { train_steps };

        let t0 = Instant::now();
        let loss = trainer.step_iteration(traversals, steps, batch, &mut rng);
        let loss_str = if warm {
            "warmup (chỉ thu thập mẫu)".to_string()
        } else {
            format!("Loss {:.5}", loss)
        };
        println!(
            "   📍 Iter {:03}/{} | Adv: {:7} | Strat: {:7} | {} | {:.1}s",
            global, total, trainer.adv_memory.len(), trainer.strategy_memory.len(),
            loss_str, t0.elapsed().as_secs_f64()
        );

        if !warm && k % bench_every == 0 {
            let bt = Instant::now();
            println!("\n   ⚔️  ĐẤU TRƯỜNG 8-MAX: Challenger (Iter {}) vs Champion...", global);

            let challenger_policy = DeepCFRPolicy::new(Arc::clone(&trainer.advantage_net), false);
            let champion_policy = DeepCFRPolicy::new(Arc::new(RwLock::new(current_champion_net.clone())), false);

            let arena_res = evaluate_against_champion(
                &challenger_policy,
                &champion_policy,
                arena_hands,
                arena_stack_bb,
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
                current_champion_net = trainer.advantage_net.read().unwrap().clone();
                let _ = trainer.save_models(&best_path, &best_strat_path);
                if out == "deep_cfr" {
                    let _ = trainer.save_models("deep_cfr_model.json", "deep_cfr_strategy.json");
                    println!("   🌟 Đã cập nhật trực tiếp vào deep_cfr_model.json & deep_cfr_strategy.json cho Receiver!");
                }
            } else {
                println!("   🛡️  Champion giữ vững ngôi vương. Challenger chưa đủ vượt trội (hoặc TrashShove cao).");
            }

            let r = bench_policy(&trainer.advantage_net, None, bench_hands);
            print_table("Tham khảo đối thủ tĩnh", &r);
            let _ = append_csv(&csv, global, "adv", &r);
            let s = score(&r);
            println!("   🧪 Benchmark log score: {:+.1} bb/100\n", s);
        }

        if k % snapshot_every == 0 {
            let a = format!("{}_iter{}_model.json", out, global);
            let s = format!("{}_iter{}_strategy.json", out, global);
            let _ = trainer.save_models(&a, &s);
            println!("   💾 Snapshot iter {} -> {}", global, a);
        }
    }

    let _ = trainer.save_models(&last_adv, &last_strat);
    println!("\n   ✅ Done in {:.1}s -> {}, {}", t_all.elapsed().as_secs_f64(), last_adv, last_strat);

    // Final benchmark 10k ván trên BEST Champion
    println!("\n🏁 Final benchmark trên BEST CHAMPION (10k ván/ô)...");
    if let Ok(m) = MLP::load_from_file(&best_path) {
        let r_best = bench_policy(&Arc::new(RwLock::new(m)), None, 10_000);
        print_table("BEST CHAMPION CUỐI CÙNG", &r_best);
        let _ = append_csv(&csv, total, "final_best", &r_best);
    }
}

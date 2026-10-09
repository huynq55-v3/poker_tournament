//! Train tiếp từ checkpoint.
//!
//! cargo run --release --bin resume -- [--from F] [--from-strat F] [--start-iter N]
//!     [--iters N] [--out PREFIX] [--warmup N] [--lr X] [--bench-hands N] [--chip-ev]

use poker_tournament::benchmark::{append_csv, print_table, run_suite, score, BenchResult};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::FEATURE_DIM;
use poker_tournament::neural_network::MLP;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const HIDDEN: [usize; 3] = [128, 128, 64];
const DEPTHS: [u32; 2] = [50, 25];

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn bench_adv(net: &Arc<RwLock<MLP>>, hands: usize) -> Vec<BenchResult> {
    // Không có strategy net -> chơi bằng regret matching của advantage net
    let policy = DeepCFRPolicy::new(Arc::clone(net), false);
    run_suite(&policy, hands, &DEPTHS)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let from_adv: String = arg(&args, "--from", "deep_cfr_last_model.json".to_string());
    let from_strat: String = arg(&args, "--from-strat", "deep_cfr_last_strategy.json".to_string());
    let out: String = arg(&args, "--out", "r2".to_string());
    let start_iter: usize = arg(&args, "--start-iter", 300);
    let iters: usize = arg(&args, "--iters", 300);
    let warmup: usize = arg(&args, "--warmup", 5);
    let lr: f32 = arg(&args, "--lr", 0.0005);
    let bench_hands: usize = arg(&args, "--bench-hands", 4000);
    let use_icm = !has(&args, "--chip-ev");

    let traversals = 400;
    let train_steps = 200;
    let batch = 512;
    let bench_every = 10;
    let snapshot_every = 50;

    let csv = format!("{}_benchmark.csv", out);
    let best_path = format!("{}_best_model.json", out);
    let last_adv = format!("{}_last_model.json", out);
    let last_strat = format!("{}_last_strategy.json", out);

    println!("============================================================");
    println!("🔁 RESUME DEEP CFR TRAINING");
    println!("============================================================");

    let _ = poker_tournament::equity::preflop_table();

    let mut rng = rand::thread_rng();
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &HIDDEN, 1_000_000);
    trainer.fixed_players = Some(6);
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
    println!(
        "   ⚙️  start_iter={} iters={} warmup={} lr={} reward={} bench_hands={}",
        start_iter, iters, warmup, lr, if use_icm { "ICM" } else { "chip-EV" }, bench_hands
    );

    // Mốc ban đầu: phải gần kết quả của iter 300 lần trước, nếu không là nạp sai
    let base = bench_adv(&trainer.advantage_net, bench_hands);
    print_table("MỐC BAN ĐẦU sau khi nạp (advantage net)", &base);
    let _ = append_csv(&csv, start_iter, "adv", &base);
    let mut best = score(&base);
    if let Ok(n) = trainer.advantage_net.read() {
        let _ = n.save_to_file(&best_path);
    }

    let total = start_iter + iters;
    let t_all = Instant::now();
    for k in 1..=iters {
        let global = start_iter + k;
        let warm = k <= warmup;
        let steps = if warm { 0 } else { train_steps };

        let t0 = Instant::now();
        let loss = trainer.step_iteration(traversals, steps, batch, &mut rng);
        let loss_str = if warm {
            "warmup (chỉ thu thập)".to_string()
        } else {
            format!("Loss {:.5}", loss)
        };
        println!(
            "   📍 Iter {:03}/{} | Adv: {:7} | Strat: {:7} | {} | {:.1}s",
            global, total, trainer.adv_memory.len(), trainer.strategy_memory.len(),
            loss_str, t0.elapsed().as_secs_f64()
        );

        if !warm && k % bench_every == 0 {
            let r = bench_adv(&trainer.advantage_net, bench_hands);
            print_table("advantage net", &r);
            let _ = append_csv(&csv, global, "adv", &r);
            let s = score(&r);
            if s > best {
                best = s;
                if let Ok(n) = trainer.advantage_net.read() {
                    let _ = n.save_to_file(&best_path);
                }
                println!("   ⭐ New best ({:+.1}) -> {}", s, best_path);
            }
        }

        if k % snapshot_every == 0 {
            let a = format!("{}_iter{}_model.json", out, global);
            let s = format!("{}_iter{}_strategy.json", out, global);
            let _ = trainer.save_models(&a, &s);
            println!("   💾 Snapshot iter {} -> {}", global, a);

            // Chỉ để theo dõi: strategy net có đuổi kịp advantage net không
            if trainer.strategy_trained {
                let p = trainer.get_policy(false);
                let r = run_suite(&p, bench_hands, &DEPTHS);
                print_table("average strategy (theo dõi)", &r);
                let _ = append_csv(&csv, global, "avg", &r);
            }
        }
    }

    let _ = trainer.save_models(&last_adv, &last_strat);
    println!("\n   ✅ Done in {:.1}s -> {}, {}", t_all.elapsed().as_secs_f64(), last_adv, last_strat);

    println!("\n🏁 Final benchmark (20k ván / ô)");
    let r_last = bench_adv(&trainer.advantage_net, 20_000);
    print_table("LAST", &r_last);
    let _ = append_csv(&csv, total, "final_last", &r_last);

    if let Ok(m) = MLP::load_from_file(&best_path) {
        let r_best = bench_adv(&Arc::new(RwLock::new(m)), 20_000);
        print_table("BEST", &r_best);
        let _ = append_csv(&csv, total, "final_best", &r_best);
    }

    println!(
        "\n➡️  Train tiếp: cargo run --release --bin resume -- --from {} --from-strat {} --start-iter {} --out {}b",
        last_adv, last_strat, total, out
    );
}

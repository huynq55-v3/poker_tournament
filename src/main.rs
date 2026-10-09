use poker_tournament::benchmark::{append_csv, print_table, run_suite, score};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::{Environment, FEATURE_DIM};
use poker_tournament::gui_bridge::TableViewModel;
use poker_tournament::neural_network::MLP;
use poker_tournament::sampler::TournamentStackSampler;
use poker_tournament::self_play::SelfPlayEngine;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const BEST_ADV: &str = "deep_cfr_model.json";
const BEST_STRAT: &str = "deep_cfr_strategy.json";
const LAST_ADV: &str = "deep_cfr_last_model.json";
const LAST_STRAT: &str = "deep_cfr_last_strategy.json";
const BENCH_CSV: &str = "benchmark_log.csv";

fn main() {
    println!("============================================================");
    println!("🚀 POKER AI DEEP CFR (PARALLEL TRAINING + LIVE BENCHMARK) 🚀");
    println!("============================================================");

    let mut rng = rand::thread_rng();

    println!("\n[0] 📐 Preparing preflop equity table...");
    let t0 = Instant::now();
    let _ = poker_tournament::equity::preflop_table();
    println!("   ✅ Ready in {:.1}s", t0.elapsed().as_secs_f64());

    // ---------------------------------------------------------------- Training
    println!("\n[1] 🏋️ Deep CFR training with live benchmark...");
    let hidden_dims = [128, 128, 64];
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, 1_000_000);
    trainer.fixed_players = Some(6);
    trainer.use_icm = false; // chip-EV giai đoạn đầu

    let total_iterations = 300;
    let icm_from_iter = 200;
    let traversals_per_iter = 400;
    let train_steps_per_iter = 200;
    let batch_size = 512;

    // Benchmark
    let bench_every = 10;
    let bench_hands = 2000; // mỗi ô (đối thủ x độ sâu)
    let depths = [50u32, 25u32];
    let _ = std::fs::remove_file(BENCH_CSV);
    let mut best_score = f64::NEG_INFINITY;

    let train_start = Instant::now();
    for iter in 1..=total_iterations {
        if iter == icm_from_iter {
            trainer.use_icm = true;
            println!("   🔄 Switching reward: chip-EV -> ICM");
        }

        let it0 = Instant::now();
        let adv_loss = trainer.step_iteration(traversals_per_iter, train_steps_per_iter, batch_size, &mut rng);
        println!(
            "   📍 Iter {:03}/{} | Adv: {:7} | Strat: {:7} | Loss: {:.5} | {:.1}s",
            iter, total_iterations,
            trainer.adv_memory.len(), trainer.strategy_memory.len(),
            adv_loss, it0.elapsed().as_secs_f64()
        );

        if iter == 1 || iter % bench_every == 0 {
            let bt = Instant::now();

            // (a) Regret-matching từ advantage net hiện tại
            let adv_only = DeepCFRPolicy::new(Arc::clone(&trainer.advantage_net), false);
            let r_adv = run_suite(&adv_only, bench_hands, &depths);
            print_table("advantage net (regret matching)", &r_adv);
            let _ = append_csv(BENCH_CSV, iter, "adv", &r_adv);

            // (b) Policy dùng để chơi (average strategy nếu đã train, nếu không thì như (a))
            let (play_label, play_results) = ("adv", r_adv);

            let s = score(&play_results);
            println!("   🧪 Benchmark ({}) took {:.1}s | score {:+.1} bb/100 | best {:+.1}",
                play_label, bt.elapsed().as_secs_f64(), s,
                if best_score.is_finite() { best_score } else { s });

            if s > best_score {
                best_score = s;
                match trainer.save_models(BEST_ADV, BEST_STRAT) {
                    Ok(_) => println!("   ⭐ New best ({:+.1} bb/100) -> saved {} / {}", s, BEST_ADV, BEST_STRAT),
                    Err(e) => println!("   ⚠️ Save failed: {}", e),
                }
            }
        }
    }

    println!("   ✅ Training complete in {:.1}s", train_start.elapsed().as_secs_f64());
    let _ = trainer.save_models(LAST_ADV, LAST_STRAT);

    // ---------------------------------------------------------------- Load best + final benchmark
    println!("\n[2] 🏁 Final benchmark on BEST checkpoint...");
    let adv = MLP::load_from_file(BEST_ADV).unwrap_or_else(|_| trainer.advantage_net.read().unwrap().clone());
    let strat = MLP::load_from_file(BEST_STRAT).ok();
    let best_policy = None;

    let final_res = run_suite(best_policy.as_ref(), 20_000, &depths);
    print_table("BEST checkpoint (20k hands/cell)", &final_res);
    let _ = append_csv(BENCH_CSV, total_iterations, "final_best", &final_res);

    // ---------------------------------------------------------------- Inspect
    println!("\n[3] 🎯 Action probabilities on a 6-player hand...");
    let num_players = 6;
    let (bb, sb, button_idx) = (100, 50, 0);
    let stacks = TournamentStackSampler::sample_dirichlet_stacks(&mut rng, num_players, bb);
    let mut env = Environment::new(num_players, &stacks, bb, sb, button_idx);
    env.reset_hand(&mut rng, button_idx);
    let mask = env.get_action_mask();
    let table_vm = TableViewModel::from_game_state(&env.state, &mask, Some(&*best_policy), true);
    println!("{}", serde_json::to_string_pretty(&table_vm).unwrap());

    // ---------------------------------------------------------------- Throughput
    println!("\n[4] ⚡️ Self-play throughput...");
    let engine = SelfPlayEngine::new();
    let num_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(4);
    let sim_start = Instant::now();
    let rx = engine.run_parallel_simulation(best_policy, num_workers, 20_000, 10_000);
    let mut total_transitions = 0u64;
    while let Ok(_t) = rx.recv() {
        total_transitions += 1;
    }
    let elapsed = sim_start.elapsed().as_secs_f64();
    let simulated = engine.total_simulated_hands.load(std::sync::atomic::Ordering::Relaxed);
    println!("   🃏 {} hands, {} transitions, {:.1} hands/sec", simulated, total_transitions, simulated as f64 / elapsed);
    println!("\n📈 Benchmark history: {}", BENCH_CSV);
}

use poker_tournament::benchmark::{
    append_csv, evaluate_against_champion, print_table, run_suite, score,
};
use poker_tournament::deep_cfr::DeepCFRPolicy;
use poker_tournament::deep_cfr::DeepCFRTrainer;
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
    println!("🚀 POKER AI DEEP CFR (8-MAX ARENA GATING / 500 ITERATIONS) 🚀");
    println!("============================================================");

    let mut rng = rand::thread_rng();

    println!("\n[0] 📐 Preparing preflop equity table...");
    let t0 = Instant::now();
    let _ = poker_tournament::equity::preflop_table();
    println!("   ✅ Ready in {:.1}s", t0.elapsed().as_secs_f64());

    println!("\n[1] 🏋️ Deep CFR training with 8-Max Arena Gating...");
    let hidden_dims = [256, 256, 128];
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, 2_000_000);
    trainer.fixed_players = None; // Trainer tự ưu tiên 70% bàn 6-8 max trong trainer.rs

    let total_iterations = 500;
    let traversals_per_iter = 400;
    let train_steps_per_iter = 200;
    let batch_size = 512;

    let bench_every = 10;
    let arena_hands = 6000;
    let arena_stack_bb = 50;
    let bench_hands = 2000;
    let depths = [50u32, 25u32];

    let _ = std::fs::remove_file(BENCH_CSV);

    let mut current_champion_net: MLP = trainer.advantage_net.read().unwrap().clone();
    let _ = trainer.save_models(BEST_ADV, BEST_STRAT);

    let train_start = Instant::now();
    for iter in 1..=total_iterations {
        let it0 = Instant::now();
        let adv_loss = trainer.step_iteration(traversals_per_iter, train_steps_per_iter, batch_size, &mut rng);
        println!(
            "   📍 Iter {:03}/{} | Adv: {:7} | Strat: {:7} | Loss: {:.5} | {:.1}s",
            iter, total_iterations,
            trainer.adv_memory.len(), trainer.strategy_memory.len(),
            adv_loss, it0.elapsed().as_secs_f64()
        );

        if iter % bench_every == 0 {
            let bt = Instant::now();

            println!("\n   ⚔️  ĐẤU TRƯỜNG 8-MAX: Challenger (Iter {}) vs Champion...", iter);
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
                    arena_res.challenger_bb_per_100, BEST_ADV
                );
                current_champion_net = trainer.advantage_net.read().unwrap().clone();
                let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
            } else {
                println!("   🛡️  Champion giữ vững ngôi vương. Chưa đủ cách biệt thống kê.");
            }

            let r_adv = run_suite(&challenger_policy, bench_hands, &depths);
            print_table("Tham khảo đối thủ tĩnh", &r_adv);
            let _ = append_csv(BENCH_CSV, iter, "adv", &r_adv);

            let s = score(&r_adv);
            println!("   🧪 Benchmark log score: {:+.1} bb/100\n", s);
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

    println!("\n[3] 🎯 Action probabilities on an 8-player hand...");
    let num_players = 8;
    let (bb, sb, button_idx) = (100, 50, 0);
    let stacks = TournamentStackSampler::sample_dirichlet_stacks(&mut rng, num_players, bb);
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

use poker_tournament::deep_cfr::DeepCFRTrainer;
use poker_tournament::environment::Environment;
use poker_tournament::gui_bridge::TableViewModel;
use poker_tournament::sampler::TournamentStackSampler;
use poker_tournament::self_play::SelfPlayEngine;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("🚀 POKER AI DEEP CFR (CPU MULTI-THREADED TRAINING & SIM) 🚀");
    println!("============================================================");

    let mut rng = rand::thread_rng();

    // 1. Hardware Architecture & Recommendation
    println!("\n[1] 🧠 Hardware Selection & Architecture Rationale:");
    println!("   • CPU: Host Multi-core CPU + 32GB RAM (AVX2/SIMD + Rayon)");
    println!("   • GPU: Intel Iris Xe (Integrated GPU - Skipped)");
    println!("   💡 Rationale: In MCCFR tree traversals, batch size per tree node is 1 (latency-sensitive).");
    println!("     GPU kernel dispatch overhead (~10-50µs) on Iris Xe is 200x slower than CPU L1/L2 cache (~80ns).");
    println!("     Pure Rust CPU execution provides maximum throughput and zero external dependency friction!");

    // 2. Training Deep CFR on CPU
    println!("\n[2] 🏋️ Starting Deep CFR Training Iterations...");
    let input_dim = 126;
    let hidden_dims = [256, 256, 128];
    let mut trainer = DeepCFRTrainer::new(input_dim, &hidden_dims, 50_000, 50_000);

    let total_iterations = 10;
    let traversals_per_iter = 25;
    let train_steps_per_iter = 50;
    let batch_size = 128;

    let train_start = Instant::now();

    for iter in 1..=total_iterations {
        let iter_start = Instant::now();
        let (adv_loss, strat_loss) = trainer.step_iteration(
            traversals_per_iter,
            train_steps_per_iter,
            batch_size,
            &mut rng,
        );
        let iter_time = iter_start.elapsed().as_secs_f64();
        println!(
            "   📍 Iteration {:02}/{} | Adv Memory: {:5} | Strat Memory: {:5} | Adv Loss: {:.5} | Strat Loss: {:.5} | ({:.2}s)",
            iter,
            total_iterations,
            trainer.adv_memory.len(),
            trainer.strat_memory.len(),
            adv_loss,
            strat_loss,
            iter_time
        );
    }

    let total_train_time = train_start.elapsed().as_secs_f64();
    println!("   ✅ Deep CFR Training Complete in {:.2}s!", total_train_time);

    // 3. Inspect Trained Policy Decisions & GTO Real-time Probs
    println!("\n[3] 🎯 Inspecting GTO Action Probabilities on a 6-Player Tournament Hand...");
    let trained_policy = Arc::new(trainer.get_policy(false));

    let num_players = 6;
    let bb = 100;
    let sb = 50;
    let button_idx = 0;
    let stacks = TournamentStackSampler::sample_dirichlet_stacks(&mut rng, num_players, bb);

    let mut env = Environment::new(num_players, &stacks, bb, sb, button_idx);
    env.reset_hand(&mut rng, button_idx);

    let mask = env.get_action_mask();
    let table_vm = TableViewModel::from_game_state(&env.state, &mask, Some(&*trained_policy), true);
    println!("📋 Table State & GTO Probabilities (JSON):\n{}", serde_json::to_string_pretty(&table_vm).unwrap());

    // 4. Multi-Threaded Self-Play Benchmark with Trained Deep CFR Policy
    println!("\n[4] ⚡️ Benchmarking Trained Deep CFR Policy in Multi-Threaded Rayon Simulation...");
    let engine = SelfPlayEngine::new();
    let num_workers = num_cpus::get().max(4);
    let target_hands: u64 = 20_000;
    let buffer_capacity = 10_000;

    println!("🔥 Running {} parallel workers evaluating Deep CFR NN inference...", num_workers);
    let sim_start = Instant::now();
    let rx = engine.run_parallel_simulation(trained_policy, num_workers, target_hands, buffer_capacity);

    let mut total_transitions = 0u64;
    while let Ok(_transition) = rx.recv() {
        total_transitions += 1;
    }

    let elapsed = sim_start.elapsed().as_secs_f64();
    let simulated = engine.total_simulated_hands.load(std::sync::atomic::Ordering::Relaxed);
    let hands_per_sec = (simulated as f64) / elapsed;

    println!("\n============================================================");
    println!("🏁 DEEP CFR MULTI-THREADED BENCHMARK FINISHED!");
    println!("   ⏱️ Simulation Time: {:.2}s", elapsed);
    println!("   🃏 Hands Simulated: {}", simulated);
    println!("   📦 Transitions Generated: {}", total_transitions);
    println!("   🚀 Throughput with NN Inference: {:.1} Hands/sec", hands_per_sec);
    println!("============================================================");
}

mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

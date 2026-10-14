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
    println!("🚀 POKER AI DEEP CFR (8-MAX ARENA GATING / 10,000 ITERATIONS) 🚀");
    println!("============================================================");

    let mut rng = rand::thread_rng();

    println!("\n[0] 📐 Preparing preflop equity table...");
    let t0 = Instant::now();
    let _ = poker_tournament::equity::preflop_table();
    println!("   ✅ Ready in {:.1}s", t0.elapsed().as_secs_f64());

    println!("\n[1] 🏋️ Deep CFR training with 8-Max Arena Gating & Tournament ICM...");
    
    // Kiến trúc mạng nơ-ron: 3 tầng ẩn đủ sâu cho 176 features
    let hidden_dims = [256, 256, 128];
    
    // Capacity 1.5M - 2M mẫu là vừa vặn bộ nhớ RAM
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, 1_500_000, true);
    trainer.fixed_players = None; // Tiếp tục ưu tiên bàn 6-8 max
    trainer.use_icm = true;       // 🟢 BẮT BUỘC: Bật hàm thưởng Tournament ICM
    trainer.lr = 0.0005;          // 🟢 Giảm nhẹ learning rate để tránh dao động All-in rác

    let total_iterations = 10_000;    // 🚀 Chạy liên tục 10,000 iterations
    let traversals_per_iter = 250;    // 🟢 Duyệt cây mỗi iter (nhờ Equity Cache nên chạy siêu tốc)
    let train_steps_per_iter = 400;   // 🟢 Ép mạng học kỹ các tình huống ICM
    let batch_size = 512;             // Giữ nguyên 512 (Kích thước vàng cho SIMD/Rayon)

    let bench_every = 10;             // Benchmark và đấu trường Arena mỗi 10 iters
    let arena_hands = 6000;           // Đủ tin cậy để soán ngôi Champion
    let arena_stack_bb = 35;          // Mức stack thực chiến phổ biến nhất của SNG
    let bench_hands = 6000;           // Đánh giá đối thủ tĩnh
    let depths = [50u32, 25u32, 12u32]; // BẮT BUỘC có 12 BB để kiểm tra chế độ Push/Fold sinh tử!

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

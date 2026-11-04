//! 🚀 DEEP CFR LOUNGE CASH GAME TRAINER (REAL CHIP / CHIP-EV / 12-300+ BB)
//!
//! Chương trình huấn luyện chuyên dụng cho chế độ Lounge PokerNow / Real Cash Game:
//! - Bàn chơi tiền thật (Lounge / Ring Game): Không có cơ chế giải đấu Tournament hay loại dần.
//! - Hàm thưởng thuần Chip-EV (không dùng Tournament ICM, mỗi 1 chip = tiền thật).
//! - Phân phối Stack cực kỳ phong phú và biến thiên ngẫu nhiên giữa các người chơi:
//!   • Min Buy-in: 12 BB (Short-stackers).
//!   • Standard Buy-in: 40 - 100 BB.
//!   • Deep-stack tích lũy: 100 - 300+ BB (không rút chip ra được, hỗ trợ dải mượt hiệu quả tới 300+ BB).
//! - Hỗ trợ quy mô bàn linh hoạt (tối đa 8 người chuẩn kiến trúc feature vector, mặc định 8-max).
//! - Kiến trúc mạng nơ-ron tinh gọn: [128, 128, 128] giảm mạnh số parameter để train siêu nhanh!
//! - Độc lập hoàn toàn: Lưu ra file `deep_cfr_lounge_model.json` và `deep_cfr_lounge_strategy.json`.

use poker_tournament::benchmark::{
    evaluate_against_champion_cash, ArenaMatchResult,
};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::FEATURE_DIM;
use poker_tournament::neural_network::MLP;
use rand::thread_rng;
use std::sync::{Arc, RwLock};
use std::time::Instant;

const BEST_ADV: &str = "deep_cfr_lounge_model.json";
const BEST_STRAT: &str = "deep_cfr_lounge_strategy.json";
const LAST_ADV: &str = "deep_cfr_lounge_last_model.json";
const LAST_STRAT: &str = "deep_cfr_lounge_last_strategy.json";
const BENCH_CSV: &str = "benchmark_lounge_log.csv";

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn print_help() {
    println!("============================================================");
    println!("🚀 POKER AI DEEP CFR - LOUNGE CASH GAME TRAINER 🚀");
    println!("============================================================");
    println!("Cú pháp: cargo run --release --bin train_lounge -- [TÙY CHỌN]");
    println!();
    println!("Tùy chọn huấn luyện:");
    println!("  --players <N>         Quy mô bàn tối đa: 2 đến 8 người (mặc định: 8)");
    println!("  --iters <N>           Số iteration (mặc định: 10,000)");
    println!("  --traversals <N>      Số ván duyệt cây mỗi iter (mặc định: 250)");
    println!("  --train-steps <N>     Số bước SGD train mạng mỗi iter (mặc định: 400)");
    println!("  --batch-size <N>      Batch size SGD (mặc định: 512)");
    println!("  --lr <F>              Learning rate cơ sở (mặc định: 0.0005)");
    println!("  --hidden <N>          Kích thước mỗi tầng ẩn (mặc định: 128 để train siêu nhanh)");
    println!("  --sparring            Bật đấu tập với 8 trường phái Quân Xanh (mặc định: TẮT)");
    println!("  --fresh               Bắt đầu train mạng mới hoàn toàn (bỏ qua model cũ)");
    println!();
    println!("Tùy chọn Đấu trường Arena:");
    println!("  --bench-every <N>     Tần suất đấu trường Arena kiểm tra Champion (mặc định: mỗi 10 iters)");
    println!("  --arena-hands <N>     Số ván đấu trường Arena đối đầu Champion (mặc định: 6,000)");
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

    let max_players: usize = arg(&args, "--players", arg(&args, "--max-players", 8usize)).clamp(2, 8);
    let iters: usize = arg(&args, "--iters", 10_000usize);
    let traversals: usize = arg(&args, "--traversals", 250usize);
    let train_steps: usize = arg(&args, "--train-steps", 400usize);
    let batch_size: usize = arg(&args, "--batch-size", 512usize);
    let bench_every: usize = arg(&args, "--bench-every", 10usize);
    let arena_hands: usize = arg(&args, "--arena-hands", 6000usize);
    let lr: f32 = arg(&args, "--lr", 0.0005f32);
    let hidden_size: usize = arg(&args, "--hidden", 128usize);
    let enable_sparring = args.iter().any(|a| a == "--sparring" || a == "--spar" || a == "--with-sparring");
    let fresh = args.iter().any(|a| a == "--fresh" || a == "--new" || a == "--from-scratch");

    println!("============================================================");
    println!("🔥 DEEP CFR LOUNGE CASH GAME TRAINER (REAL CHIP / CHIP-EV) 🔥");
    println!("============================================================");
    println!("  • Quy mô bàn       : Bàn {}-Max Lounge (Ring Game tiền thật)", max_players);
    println!("  • Cơ chế thưởng    : Thuần Chip-EV (Mỗi chip = Tiền thật, không ICM)");
    println!("  • Phân bổ Stack    : Random ngẫu nhiên từng ghế (Min 12 BB, Standard 40-100 BB, Deep 100-300+ BB)");
    println!("  • Feature Scaling  : Cơ số chuẩn 150 BB (Dải mượt 0..300+ BB không kịch trần)");
    println!("  • Cấu trúc Mạng    : [{0}, {0}, {0}] ({1} features, tinh gọn siêu tốc)", hidden_size, FEATURE_DIM);
    println!("  • Quân Xanh        : {}", if enable_sparring { "BẬT (50% Đấu Tập Quân Xanh / 50% Self-Play)" } else { "TẮT (100% Thuần Self-Play)" });
    println!("  • File Output      : {} & {}", BEST_ADV, BEST_STRAT);
    println!("============================================================");

    let hidden_dims = [hidden_size, hidden_size, hidden_size];
    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &hidden_dims, 3_000_000, false);
    trainer.is_cash_game = true;
    trainer.max_players = max_players;
    trainer.fixed_players = Some(max_players);
    trainer.use_icm = false;
    trainer.stack_bb = 0; // Kích hoạt nhánh Cash/Lounge Stack Random (Min 12 BB -> Deep 300+ BB)
    trainer.enable_sparring = enable_sparring;
    trainer.lr = lr;

    let from_adv: String = arg(&args, "--from", BEST_ADV.to_string());
    let from_strat: String = arg(&args, "--from-strat", BEST_STRAT.to_string());

    if !fresh && std::path::Path::new(&from_adv).exists() {
        println!("📥 Đang nạp checkpoint có sẵn: {}", from_adv);
        match trainer.load_models(&from_adv, &from_strat) {
            Ok(true) => println!("   ✅ Nạp thành công Advantage Net và Strategy Net!"),
            Ok(false) => println!("   ✅ Nạp Advantage Net (Strategy Net khởi tạo mới)!"),
            Err(e) => println!("   ⚠️  Không khớp kiến trúc ({}), bắt đầu train mạng mới hoàn toàn.", e),
        }
    } else {
        println!("🌱 Khởi tạo mạng mới hoàn toàn cho Lounge Cash Game (Zero-bias).");
        let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
    }

    let mut current_champion_net: MLP = trainer.advantage_net.read().unwrap().clone();

    let mut rng = thread_rng();
    let train_start = Instant::now();

    for iter in 1..=iters {
        let it0 = Instant::now();
        let adv_loss = trainer.step_iteration(traversals, train_steps, batch_size, &mut rng);
        println!(
            "   📍 Lounge Iter {:05}/{} | Adv: {:7} | Strat: {:7} | Loss: {:.5} | {:.1}s",
            iter, iters,
            trainer.adv_memory.len(), trainer.strategy_memory.len(),
            adv_loss, it0.elapsed().as_secs_f64()
        );

        if iter % bench_every == 0 {
            let bt = Instant::now();
            println!("\n   ⚔️  ĐẤU TRƯỜNG LOUNGE {}-MAX (Chip-EV, 12-300+ BB): Challenger (Iter {}) vs Champion...", max_players, iter);
            let challenger_policy = trainer.get_policy(false);
            let champion_strat = if std::path::Path::new(BEST_STRAT).exists() {
                MLP::load_from_file(BEST_STRAT).ok()
            } else {
                None
            };
            let champion_policy = DeepCFRPolicy::with_strategy(
                Arc::new(RwLock::new(current_champion_net.clone())),
                champion_strat,
                false,
            );

            let arena_res: ArenaMatchResult = evaluate_against_champion_cash(
                &challenger_policy,
                &champion_policy,
                arena_hands,
                max_players,
            );

            println!(
                "   📊 Kết quả Arena Lounge: {:+.2} ± {:.2} bb/100 | TrashShove: {:.2}% | {} ván trong {:.1}s",
                arena_res.challenger_bb_per_100,
                arena_res.ci95,
                arena_res.challenger_trash_shove_pct,
                arena_res.hands,
                bt.elapsed().as_secs_f64()
            );

            if arena_res.is_winner {
                println!(
                    "   👑 SOÁN NGÔI CHAMPION! Iter {} thắng vượt trội {:+.2} bb/100 -> Đã cập nhật {}!",
                    iter, arena_res.challenger_bb_per_100, BEST_ADV
                );
                current_champion_net = trainer.advantage_net.read().unwrap().clone();
                let _ = trainer.save_models(BEST_ADV, BEST_STRAT);
            } else {
                println!("   🛡️  Champion vẫn giữ vững vị thế.");
            }

            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(BENCH_CSV) {
                let _ = writeln!(
                    f, "{},lounge_arena,{:.2},{:.2},{:.2},{}", 
                    iter, 
                    arena_res.challenger_bb_per_100, 
                    arena_res.ci95, 
                    arena_res.challenger_trash_shove_pct, 
                    arena_res.hands
                );
            }

            // Luôn lưu checkpoint gần nhất
            let _ = trainer.save_models(LAST_ADV, LAST_STRAT);
            println!();
        }
    }

    println!("🎉 Hoàn thành huấn luyện Lounge Cash Game trong {:.1} phút!", train_start.elapsed().as_secs_f64() / 60.0);
}

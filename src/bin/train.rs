use poker_tournament::benchmark::{
    append_csv, evaluate_against_champion_cash, evaluate_against_champion_tournament, print_table,
    run_cash_suite, run_suite, score, simulate_demo_tournament, simulate_hero_tournament,
    TightAggressive,
};
use poker_tournament::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use poker_tournament::environment::{FEATURE_DIM, MAX_PLAYERS};
use poker_tournament::equity::preflop_table;
use poker_tournament::neural_network::MLP;
use poker_tournament::self_play::Policy;
use rand::thread_rng;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TrainMode {
    Tournament,
    ChipEv,
}

impl TrainMode {
    fn label(self) -> &'static str {
        match self {
            TrainMode::Tournament => "TOURNAMENT (ICM + MULTI-DEPTH)",
            TrainMode::ChipEv => "CHIP-EV / LOUNGE CASH (12-300+ BB)",
        }
    }
}

#[derive(Clone, Debug)]
struct TrainConfig {
    mode: TrainMode,
    players: usize,
    fixed_players: Option<usize>,
    stack_bb: u32,
    use_icm: bool,
    is_cash_game: bool,
    sparring: bool,
    fresh: bool,
    from_adv: Option<String>,
    from_strat: Option<String>,
    champ_adv: Option<String>,
    out_prefix: String,
    start_iter: Option<usize>,
    iters: usize,
    warmup: usize,
    traversals: usize,
    train_steps: usize,
    batch_size: usize,
    lr: f32,
    hidden: Vec<usize>,
    capacity: usize,
    bench_every: usize,
    arena_tours: usize,
    arena_hands: usize,
    bench_hands: usize,
    snapshot_every: usize,
    final_sim: bool,
}

impl TrainConfig {
    fn for_mode(mode: TrainMode, fresh: bool) -> Self {
        match mode {
            TrainMode::Tournament => Self {
                mode,
                players: 8,
                fixed_players: None,
                stack_bb: 50,
                use_icm: true,
                is_cash_game: false,
                sparring: false,
                fresh,
                from_adv: None,
                from_strat: None,
                champ_adv: None,
                out_prefix: "deep_cfr".to_string(),
                start_iter: None,
                iters: if fresh { 10_000 } else { 500 },
                warmup: if fresh { 0 } else { 5 },
                traversals: 400,
                train_steps: 128,
                batch_size: 4096,
                lr: 1e-4,
                hidden: vec![512, 256, 128],
                capacity: 5_000_000,
                bench_every: 10,
                arena_tours: 100,
                arena_hands: 4000,
                bench_hands: 4000,
                snapshot_every: 100,
                final_sim: false,
            },
            TrainMode::ChipEv => Self {
                mode,
                players: MAX_PLAYERS,
                fixed_players: None,
                stack_bb: 0, // 0 = ngẫu nhiên 12-300+ BB (Lounge Cash)
                use_icm: false,
                is_cash_game: true,
                sparring: false,
                fresh,
                from_adv: None,
                from_strat: None,
                champ_adv: None,
                out_prefix: "deep_cfr_lounge".to_string(),
                start_iter: None,
                iters: if fresh { 10_000 } else { 500 },
                warmup: if fresh { 0 } else { 5 },
                traversals: 400,
                train_steps: 128,
                batch_size: 4096,
                lr: 1e-4,
                hidden: vec![512, 256, 128],
                capacity: 3_000_000,
                bench_every: 10,
                arena_tours: 100,
                arena_hands: 6000,
                bench_hands: 2000,
                snapshot_every: 100,
                final_sim: false,
            },
        }
    }

    fn champ_adv_path(&self) -> String {
        format!("{}_model.json", self.out_prefix)
    }

    fn champ_strat_path(&self) -> String {
        format!("{}_strategy.json", self.out_prefix)
    }

    fn last_adv_path(&self) -> String {
        format!("{}_last_model.json", self.out_prefix)
    }

    fn last_strat_path(&self) -> String {
        format!("{}_last_strategy.json", self.out_prefix)
    }

    fn csv_path(&self) -> String {
        match self.out_prefix.as_str() {
            "deep_cfr" => "benchmark_log.csv".to_string(),
            "deep_cfr_lounge" => "benchmark_lounge_log.csv".to_string(),
            other => format!("{}_benchmark.csv", other),
        }
    }
}

fn print_help() {
    println!("Antigravity Deep CFR Unified Trainer (Tournament ICM & Lounge Cash Chip-EV)");
    println!("Sử dụng: cargo run --release --bin train -- [OPTIONS]\n");
    println!("CHẾ ĐỘ HUẤN LUYỆN:");
    println!("  --mode <tournament|chip-ev|lounge>  Chọn chế độ (mặc định: tournament)");
    println!("  --tournament, --icm                 Tournament ICM + Multi-Depth SNG");
    println!("  --lounge, --cash, --chip-ev         Lounge Cash Chip-EV 2-10 Max");
    println!("  --no-icm                            Tắt ICM (ép về Chip-EV)");
    println!("\nBÀN CHƠI & CHECKPOINT:");
    println!("  --players <N>                       Số ghế tối đa 2..=10 (mặc định: 8 tour / 10 lounge)");
    println!("  --fixed-players                     Cố định đúng <N> ghế, không biến thiên Multi-Depth");
    println!("  --stack <BB>                        Stack cố định tính theo BB (0 = Lounge random 12-300+ BB)");
    println!("  --sparring                          Bật chế độ đấu tập 50% với Quân Xanh (bot heuristic)");
    println!("  --fresh                             Train mới hoàn toàn từ trọng số ngẫu nhiên");
    println!("  --from <path>                       Resume từ file advantage model chỉ định");
    println!("  --from-strat <path>                 Đường dẫn strategy model tương ứng (tự suy diễn nếu bỏ qua)");
    println!("  --champ <path>                      File Champion để thách đấu soán ngôi");
    println!("  --out <prefix>                      Tiền tố tên file output (mặc định: deep_cfr / deep_cfr_lounge)");
    println!("  --start-iter <N>                    Ghi đè iteration bắt đầu khi resume");
    println!("\nSIÊU THAM SỐ (HYPERPARAMETERS):");
    println!("  --iters <N>                         Số iteration cần chạy (mặc định: 10000 fresh / 500 resume)");
    println!("  --warmup <N>                        Số iter nạp đầy Replay Memory trước khi cập nhật gradient");
    println!("  --traversals <N>                    Số cây/giải traversal mỗi iter (mặc định: 400)");
    println!("  --train-steps <N>                   Số bước Adam mỗi iter (mặc định: 128)");
    println!("  --batch-size <N>                    Kích thước batch (mặc định: 4096)");
    println!("  --lr <F>                            Learning rate (mặc định: 0.0001)");
    println!("  --hidden <512 | 512,256,128>        Kiến trúc các tầng ẩn MLP (mặc định: 512,256,128)");
    println!("  --capacity <N>                      Dung lượng Reservoir Memory (mặc định: 5M tour / 3M lounge)");
    println!("\nKIỂM ĐỊNH & LƯU SNAPSHOT:");
    println!("  --bench-every <N>                   Chu kỳ chạy benchmark & Arena (mặc định: 10)");
    println!("  --arena-tours <N>                   Số giải SNG đấu Champion ở mode Tournament (mặc định: 100)");
    println!("  --arena-hands <N>                   Số ván Duplicate đấu Champion ở mode Chip-EV (mặc định: 6000)");
    println!("  --bench-hands <N>                   Số ván benchmark mỗi đối thủ (mặc định: 4000 tour / 2000 lounge)");
    println!("  --snapshot-every <N>                Lưu snapshot vào checkpoints/ mỗi N iter (0 = tắt)");
    println!("  --final-sim                         Chạy giải đấu mô phỏng cuối đợt (Tournament)");
    println!("  -h, --help                          Hiển thị hướng dẫn này");
}

fn parse_mode_value(val: &str) -> Result<TrainMode, String> {
    match val.to_ascii_lowercase().as_str() {
        "tournament" | "icm" | "sng" => Ok(TrainMode::Tournament),
        "chip-ev" | "chip_ev" | "chipev" | "lounge" | "cash" => Ok(TrainMode::ChipEv),
        other => Err(format!(
            "Chế độ '{}' không hợp lệ (hỗ trợ: tournament, chip-ev, lounge)",
            other
        )),
    }
}

fn parse_hidden_dims(raw: &str) -> Result<Vec<usize>, String> {
    if raw.contains(',') {
        let mut dims = Vec::new();
        for part in raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val = trimmed
                .parse::<usize>()
                .map_err(|_| format!("Hidden dim '{}' không hợp lệ", trimmed))?;
            if val == 0 {
                return Err("Hidden dim phải > 0".to_string());
            }
            dims.push(val);
        }
        if dims.is_empty() {
            return Err("Danh sách --hidden rỗng".to_string());
        }
        Ok(dims)
    } else {
        let n = raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Hidden dim '{}' không hợp lệ", raw))?;
        if n == 0 {
            return Err("Hidden dim phải > 0".to_string());
        }
        if n >= 256 {
            Ok(vec![n, (n / 2).max(64), (n / 4).max(64)])
        } else {
            Ok(vec![n, (n / 2).max(32)])
        }
    }
}

fn next_arg<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("Thiếu giá trị cho tham số {}", flag))
}

fn parse_train_args(args: &[String]) -> Result<Option<TrainConfig>, String> {
    // Pass 1: xác định mode và cờ --fresh trước để áp default chuẩn
    let mut mode = TrainMode::Tournament;
    let mut fresh = false;
    let mut fixed_players_flag = false;
    let mut no_icm_flag = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--mode" => {
                let val = next_arg(args, &mut i, "--mode")?;
                mode = parse_mode_value(val)?;
            }
            "--tournament" | "--icm" => mode = TrainMode::Tournament,
            "--lounge" | "--cash" | "--chip-ev" => mode = TrainMode::ChipEv,
            "--no-icm" => no_icm_flag = true,
            "--fresh" => fresh = true,
            "--fixed-players" => fixed_players_flag = true,
            _ => {}
        }
        i += 1;
    }

    let mut cfg = TrainConfig::for_mode(mode, fresh);
    if no_icm_flag {
        cfg.use_icm = false;
    }

    let mut hidden_explicit = false;

    // Pass 2: đọc toàn bộ tham số cụ thể
    let mut i = 1;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--mode" => {
                let _ = next_arg(args, &mut i, "--mode")?;
            }
            "--tournament" | "--icm" | "--lounge" | "--cash" | "--chip-ev" | "--no-icm"
            | "--fresh" => {}
            "--fixed-players" => {
                fixed_players_flag = true;
            }
            "--players" => {
                let raw = next_arg(args, &mut i, flag)?;
                let n = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--players '{}' không hợp lệ", raw))?
                    .clamp(2, MAX_PLAYERS);
                cfg.players = n;
            }
            "--stack" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.stack_bb = raw
                    .parse::<u32>()
                    .map_err(|_| format!("--stack '{}' không hợp lệ", raw))?;
            }
            "--sparring" => cfg.sparring = true,
            "--from" => cfg.from_adv = Some(next_arg(args, &mut i, flag)?.to_string()),
            "--from-strat" => cfg.from_strat = Some(next_arg(args, &mut i, flag)?.to_string()),
            "--champ" => cfg.champ_adv = Some(next_arg(args, &mut i, flag)?.to_string()),
            "--out" => cfg.out_prefix = next_arg(args, &mut i, flag)?.to_string(),
            "--start-iter" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.start_iter = Some(
                    raw.parse::<usize>()
                        .map_err(|_| format!("--start-iter '{}' không hợp lệ", raw))?,
                );
            }
            "--iters" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.iters = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--iters '{}' không hợp lệ", raw))?
                    .max(1);
            }
            "--warmup" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.warmup = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--warmup '{}' không hợp lệ", raw))?;
            }
            "--traversals" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.traversals = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--traversals '{}' không hợp lệ", raw))?
                    .max(1);
            }
            "--train-steps" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.train_steps = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--train-steps '{}' không hợp lệ", raw))?;
            }
            "--batch-size" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.batch_size = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--batch-size '{}' không hợp lệ", raw))?
                    .max(1);
            }
            "--lr" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.lr = raw
                    .parse::<f32>()
                    .map_err(|_| format!("--lr '{}' không hợp lệ", raw))?;
            }
            "--hidden" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.hidden = parse_hidden_dims(raw)?;
                hidden_explicit = true;
            }
            "--capacity" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.capacity = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--capacity '{}' không hợp lệ", raw))?
                    .max(1000);
            }
            "--bench-every" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.bench_every = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--bench-every '{}' không hợp lệ", raw))?
                    .max(1);
            }
            "--arena-tours" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.arena_tours = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--arena-tours '{}' không hợp lệ", raw))?
                    .max(1);
            }
            "--arena-hands" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.arena_hands = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--arena-hands '{}' không hợp lệ", raw))?
                    .max(2);
            }
            "--bench-hands" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.bench_hands = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--bench-hands '{}' không hợp lệ", raw))?
                    .max(10);
            }
            "--snapshot-every" => {
                let raw = next_arg(args, &mut i, flag)?;
                cfg.snapshot_every = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--snapshot-every '{}' không hợp lệ", raw))?;
            }
            "--final-sim" => cfg.final_sim = true,
            unknown => return Err(format!("Tham số không được nhận diện: '{}'", unknown)),
        }
        i += 1;
    }

    if fixed_players_flag {
        cfg.fixed_players = Some(cfg.players);
    }

    // Nếu không --fresh và chưa truyền --from, tự tìm file _last_model.json hoặc _model.json để resume
    if !cfg.fresh && cfg.from_adv.is_none() {
        let last = cfg.last_adv_path();
        let champ = cfg.champ_adv_path();
        if Path::new(&last).exists() {
            cfg.from_adv = Some(last);
        } else if Path::new(&champ).exists() {
            cfg.from_adv = Some(champ);
        } else {
            // Chưa có model nào trên đĩa -> khởi chạy fresh sạch sẽ
            cfg.warmup = 0;
        }
    }

    if let (false, Some(ref adv_path)) = (hidden_explicit, &cfg.from_adv) {
        if let Ok(peek) = MLP::load_from_file(adv_path) {
            let dims = peek.layer_dims();
            if dims.len() >= 3 {
                cfg.hidden = dims[1..dims.len() - 1].to_vec();
            }
        }
    }

    Ok(Some(cfg))
}

fn infer_strat_path(adv_path: &str) -> String {
    if let Some(stem) = adv_path.strip_suffix("_model.json") {
        format!("{}_strategy.json", stem)
    } else if let Some(stem) = adv_path.strip_suffix(".json") {
        format!("{}_strategy.json", stem)
    } else {
        format!("{}_strategy.json", adv_path)
    }
}

fn detect_start_iter(adv_path: Option<&str>, csv_path: &str) -> usize {
    if let Some(p) = adv_path {
        if let Some(idx) = p.find("_iter_") {
            let after = &p[idx + 6..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<usize>() {
                return n;
            }
        }
    }

    if let Ok(content) = std::fs::read_to_string(csv_path) {
        let mut max_iter = 0usize;
        for line in content.lines().skip(1) {
            if let Some(first) = line.split(',').next() {
                if let Ok(n) = first.trim().parse::<usize>() {
                    max_iter = max_iter.max(n);
                }
            }
        }
        return max_iter;
    }

    0
}

fn load_champion_policy(
    adv_path: &str,
    strat_path: &str,
    required_if_missing: bool,
) -> Result<Option<DeepCFRPolicy>, String> {
    if !Path::new(adv_path).exists() {
        if required_if_missing {
            return Err(format!("Không tìm thấy Champion file '{}'", adv_path));
        }
        return Ok(None);
    }

    let adv = MLP::load_from_file(adv_path)
        .map_err(|e| format!("Lỗi đọc Champion '{}': {}", adv_path, e))?;
    if adv.input_dim() != FEATURE_DIM {
        let msg = format!(
            "Champion '{}' có input_dim={} không khớp FEATURE_DIM={}",
            adv_path,
            adv.input_dim(),
            FEATURE_DIM
        );
        if required_if_missing {
            return Err(msg);
        }
        eprintln!("⚠️ {} — bỏ qua Champion cũ này.", msg);
        return Ok(None);
    }

    let strat = if Path::new(strat_path).exists() {
        match MLP::load_from_file(strat_path) {
            Ok(s) if s.input_dim() == FEATURE_DIM => Some(s),
            _ => None,
        }
    } else {
        None
    };

    Ok(Some(DeepCFRPolicy::with_strategy(
        Arc::new(RwLock::new(adv)),
        strat,
        false,
    )))
}

fn save_snapshot(trainer: &DeepCFRTrainer, out_prefix: &str, global_iter: usize) {
    let dir = Path::new("checkpoints");
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("⚠️ Không thể tạo thư mục checkpoints/: {}", e);
        return;
    }
    let adv = format!("checkpoints/{}_iter_{}_model.json", out_prefix, global_iter);
    let strat = format!("checkpoints/{}_iter_{}_strategy.json", out_prefix, global_iter);
    if let Err(e) = trainer.save_models(&adv, &strat) {
        eprintln!("⚠️ Lỗi lưu snapshot {}: {}", adv, e);
    } else {
        println!("  📦 Snapshot đã lưu -> {}", adv);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg = match parse_train_args(&args) {
        Ok(Some(c)) => c,
        Ok(None) => {
            print_help();
            return;
        }
        Err(e) => {
            eprintln!("❌ {}", e);
            eprintln!("   Dùng --help để xem hướng dẫn tham số.");
            std::process::exit(1);
        }
    };

    println!("==================================================================");
    println!("  🚀 ANTIGRAVITY DEEP CFR TRAINER [{}]", cfg.mode.label());
    println!("==================================================================");

    let start_t = Instant::now();
    print!("⏳ Đang khởi tạo bảng Preflop Equity...");
    let _ = preflop_table();
    println!(" Xong! ({:.1}s)", start_t.elapsed().as_secs_f32());

    let champ_adv_path = cfg.champ_adv_path();
    let champ_strat_path = cfg.champ_strat_path();
    let last_adv_path = cfg.last_adv_path();
    let last_strat_path = cfg.last_strat_path();
    let csv_path = cfg.csv_path();

    let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &cfg.hidden, cfg.capacity, cfg.use_icm);
    trainer.max_players = cfg.players;
    trainer.fixed_players = cfg.fixed_players;
    trainer.use_icm = cfg.use_icm;
    trainer.stack_bb = cfg.stack_bb;
    trainer.is_cash_game = cfg.is_cash_game;
    trainer.enable_sparring = cfg.sparring;
    trainer.lr = cfg.lr;

    let mut start_iter = cfg.start_iter.unwrap_or(0);
    if let Some(ref resume_adv) = cfg.from_adv {
        let resume_strat = cfg
            .from_strat
            .clone()
            .unwrap_or_else(|| infer_strat_path(resume_adv));
        println!("📥 Đang resume Advantage từ : {}", resume_adv);
        println!("📥 Đang resume Strategy  từ : {}", resume_strat);

        match trainer.load_models(resume_adv, &resume_strat) {
            Ok(has_strat) => {
                println!(
                    "   ✅ Nạp thành công (layers={:?}, strategy_trained={})",
                    cfg.hidden, has_strat
                );
                if cfg.start_iter.is_none() {
                    start_iter = detect_start_iter(Some(resume_adv), &csv_path);
                }
            }
            Err(err) => {
                eprintln!("❌ Không thể resume từ '{}': {}", resume_adv, err);
                std::process::exit(1);
            }
        }
    } else {
        println!(
            "🌱 Khởi tạo mạng mới từ đầu (layers: [{} -> {:?} -> 6])",
            FEATURE_DIM, cfg.hidden
        );
    }

    trainer.iteration = start_iter;

    let (check_champ_adv, check_champ_strat, champ_explicit) = match &cfg.champ_adv {
        Some(p) => (p.clone(), infer_strat_path(p), true),
        None => (champ_adv_path.clone(), champ_strat_path.clone(), false),
    };

    let mut champion = match load_champion_policy(&check_champ_adv, &check_champ_strat, champ_explicit) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("❌ {}", err);
            std::process::exit(1);
        }
    };

    println!(
        "  • Bàn chơi     : {}-Max (fixed={:?}) | Stack={} BB | ICM={} | Cash={}",
        cfg.players,
        cfg.fixed_players,
        if cfg.stack_bb == 0 {
            "Random 12-300+BB".to_string()
        } else {
            format!("{}BB", cfg.stack_bb)
        },
        cfg.use_icm,
        cfg.is_cash_game
    );
    println!(
        "  • Hyperparams  : iters={} (start={}), warmup={}, traversals={}, steps={}, batch={}, lr={:.1e}, sparring={}",
        cfg.iters, start_iter, cfg.warmup, cfg.traversals, cfg.train_steps, cfg.batch_size, cfg.lr, cfg.sparring
    );
    println!(
        "  • Output       : champ={}, last={}, csv={}",
        champ_adv_path, last_adv_path, csv_path
    );
    if champion.is_some() {
        println!("  👑 Đương kim Champion đang giữ đai: {}", check_champ_adv);
    } else {
        println!("  🥊 Chưa có Champion trên đĩa — thách đấu benchmark chuẩn TightAggressive để phong vương");
    }

    let mut rng = thread_rng();

    // Pha Warmup (nạp lại Reservoir Buffer khi resume từ weights sẵn có)
    if cfg.warmup > 0 && cfg.from_adv.is_some() {
        println!("\n🔥 Bắt đầu Warmup ({} iterations traversal-only)...", cfg.warmup);
        for w in 1..=cfg.warmup {
            let wt = Instant::now();
            trainer.step_iteration(cfg.traversals, 0, cfg.batch_size, &mut rng);
            println!(
                "  [Warmup {}/{}] adv_buf={}, strat_buf={} ({:.1}s)",
                w,
                cfg.warmup,
                trainer.adv_memory.len(),
                trainer.strategy_memory.len(),
                wt.elapsed().as_secs_f32()
            );
        }
    }

    println!("\n🏋️ Bắt đầu huấn luyện chính...");
    for k in 1..=cfg.iters {
        let iter_t = Instant::now();
        let adv_loss = trainer.step_iteration(
            cfg.traversals,
            cfg.train_steps,
            cfg.batch_size,
            &mut rng,
        );
        let global = trainer.iteration;

        println!(
            "[Iter {:>5} | step {:>4}/{}] loss={:.4} adv_buf={} strat_buf={} ({:.1}s)",
            global,
            k,
            cfg.iters,
            adv_loss,
            trainer.adv_memory.len(),
            trainer.strategy_memory.len(),
            iter_t.elapsed().as_secs_f32()
        );

        let should_bench = k % cfg.bench_every == 0 || k == cfg.iters;
        if should_bench {
            if let Err(e) = trainer.save_models(&last_adv_path, &last_strat_path) {
                eprintln!("⚠️ Lỗi lưu latest checkpoint: {}", e);
            }

            let policy = trainer.get_policy(false);
            match cfg.mode {
                TrainMode::Tournament => {
                    let res = run_suite(&policy, cfg.bench_hands, &[50, 25, 12, 8]);
                    print_table("GTO-Tour", &res);
                    if let Err(e) = append_csv(&csv_path, global, "GTO-Tour", &res) {
                        eprintln!("⚠️ Lỗi ghi CSV {}: {}", csv_path, e);
                    }
                    let suite_score = score(&res);

                    let opp_ref: &dyn Policy = match champion.as_ref() {
                        Some(c) => c,
                        None => &TightAggressive,
                    };
                    let tours = if champion.is_some() {
                        cfg.arena_tours
                    } else {
                        cfg.arena_tours.min(40).max(10)
                    };
                    let arena = evaluate_against_champion_tournament(
                        &policy,
                        opp_ref,
                        tours,
                        cfg.stack_bb.max(12),
                        cfg.players,
                    );
                    println!(
                        "   ⚔️ Tour Arena ({} SNGs, {}-Max): 1st {} vs {} | ITM {} | Rank {:.2} vs {:.2} | AvgICM {:.3} | TrashShove {:.1}% | Score {:+.1}",
                        arena.tournaments,
                        cfg.players,
                        arena.challenger_first_places,
                        arena.champion_first_places,
                        arena.challenger_itm_count,
                        arena.challenger_avg_rank,
                        arena.champion_avg_rank,
                        arena.challenger_avg_icm_payout,
                        arena.challenger_trash_shove_pct,
                        suite_score
                    );

                    let should_promote = arena.is_winner
                        || (champion.is_none()
                            && arena.challenger_avg_rank < arena.champion_avg_rank
                            && arena.challenger_trash_shove_pct <= 2.5);
                    if should_promote {
                        println!(
                            "   👑 CHALLENGER SOÁN NGÔI CHAMPION TOURNAMENT tại Iter {}!",
                            global
                        );
                        if let Err(e) = trainer.save_models(&champ_adv_path, &champ_strat_path) {
                            eprintln!("⚠️ Lỗi lưu Champion {}: {}", champ_adv_path, e);
                        } else {
                            champion = Some(trainer.get_policy(false));
                        }
                    }
                }
                TrainMode::ChipEv => {
                    let res = run_cash_suite(&policy, cfg.bench_hands, cfg.players);
                    print_table("Lounge-Cash", &res);
                    if let Err(e) = append_csv(&csv_path, global, "Lounge-Cash", &res) {
                        eprintln!("⚠️ Lỗi ghi CSV {}: {}", csv_path, e);
                    }
                    let suite_score = score(&res);

                    let opp_ref: &dyn Policy = match champion.as_ref() {
                        Some(c) => c,
                        None => &TightAggressive,
                    };
                    let arena = evaluate_against_champion_cash(
                        &policy,
                        opp_ref,
                        cfg.arena_hands,
                        cfg.players,
                    );
                    println!(
                        "   ⚔️ Cash Duplicate Arena ({} hands, {}-Max): {:+.2} bb/100 (±{:.2}) | TrashShove {:.1}% | Winner={} | Score {:+.1}",
                        arena.hands,
                        cfg.players,
                        arena.challenger_bb_per_100,
                        arena.ci95,
                        arena.challenger_trash_shove_pct,
                        arena.is_winner,
                        suite_score
                    );

                    let should_promote = arena.is_winner
                        || (champion.is_none()
                            && arena.challenger_bb_per_100 > 0.0
                            && arena.challenger_trash_shove_pct <= 2.5);
                    if should_promote {
                        println!(
                            "   👑 CHALLENGER SOÁN NGÔI CHAMPION LOUNGE tại Iter {}!",
                            global
                        );
                        if let Err(e) = trainer.save_models(&champ_adv_path, &champ_strat_path) {
                            eprintln!("⚠️ Lỗi lưu Champion {}: {}", champ_adv_path, e);
                        } else {
                            champion = Some(trainer.get_policy(false));
                        }
                    }
                }
            }
        }

        if cfg.snapshot_every > 0 && global % cfg.snapshot_every == 0 {
            save_snapshot(&trainer, &cfg.out_prefix, global);
        }
    }

    if let Err(e) = trainer.save_models(&last_adv_path, &last_strat_path) {
        eprintln!("⚠️ Lỗi lưu checkpoint cuối {}: {}", last_adv_path, e);
    }

    if cfg.final_sim && cfg.mode == TrainMode::Tournament {
        println!("\n🏆 Đang chạy giải đấu mô phỏng tổng kết...");
        let eval_policy = champion.unwrap_or_else(|| trainer.get_policy(false));
        let (rank, hands) = simulate_demo_tournament(
            &eval_policy,
            cfg.players,
            cfg.stack_bb.max(12),
            &mut rng,
        );
        println!("   Demo Tournament: Hero đạt hạng {} sau {} ván.", rank, hands);

        let mut top1 = 0usize;
        let mut itm = 0usize;
        let sim_tours = 100usize;
        for _ in 0..sim_tours {
            let (r, _) = simulate_hero_tournament(
                &eval_policy,
                cfg.players,
                cfg.stack_bb.max(12),
                &mut rng,
            );
            if r == 1 {
                top1 += 1;
            }
            if r <= 3 {
                itm += 1;
            }
        }
        println!(
            "   100 SNG Summary ({}-Max): Vô địch {}/{} ({:.1}%), ITM {}/{} ({:.1}%)",
            cfg.players,
            top1,
            sim_tours,
            top1 as f64 * 100.0 / sim_tours as f64,
            itm,
            sim_tours,
            itm as f64 * 100.0 / sim_tours as f64
        );
    }

    println!(
        "\n✅ Hoàn thành huấn luyện sau {:.1} phút!",
        start_t.elapsed().as_secs_f32() / 60.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_train_cli_defaults_tournament() {
        let args = to_args(&["train", "--fresh"]);
        let cfg = parse_train_args(&args).unwrap().unwrap();
        assert_eq!(cfg.mode, TrainMode::Tournament);
        assert!(cfg.use_icm);
        assert!(!cfg.is_cash_game);
        assert_eq!(cfg.players, 8);
        assert_eq!(cfg.fixed_players, None);
        assert_eq!(cfg.stack_bb, 50);
        assert_eq!(cfg.out_prefix, "deep_cfr");
        assert_eq!(cfg.capacity, 5_000_000);
        assert_eq!(cfg.csv_path(), "benchmark_log.csv");
        assert_eq!(cfg.warmup, 0);
        assert_eq!(cfg.iters, 10_000);
    }

    #[test]
    fn test_train_cli_chip_ev_lounge_flags() {
        let args = to_args(&[
            "train",
            "--mode",
            "chip-ev",
            "--fresh",
            "--players",
            "10",
            "--fixed-players",
            "--iters",
            "1200",
            "--sparring",
            "--out",
            "custom_lounge",
        ]);
        let cfg = parse_train_args(&args).unwrap().unwrap();
        assert_eq!(cfg.mode, TrainMode::ChipEv);
        assert!(!cfg.use_icm);
        assert!(cfg.is_cash_game);
        assert_eq!(cfg.players, 10);
        assert_eq!(cfg.fixed_players, Some(10));
        assert_eq!(cfg.stack_bb, 0);
        assert!(cfg.sparring);
        assert_eq!(cfg.iters, 1200);
        assert_eq!(cfg.out_prefix, "custom_lounge");
        assert_eq!(cfg.champ_adv_path(), "custom_lounge_model.json");
        assert_eq!(cfg.csv_path(), "custom_lounge_benchmark.csv");
    }

    #[test]
    fn test_train_cli_hidden_dims_formats() {
        assert_eq!(parse_hidden_dims("512").unwrap(), vec![512, 256, 128]);
        assert_eq!(parse_hidden_dims("256,256,128").unwrap(), vec![256, 256, 128]);
        assert!(parse_hidden_dims("0").is_err());
        assert!(parse_hidden_dims("abc").is_err());
    }

    #[test]
    fn test_infer_strat_and_unknown_flag() {
        assert_eq!(
            infer_strat_path("deep_cfr_lounge_last_model.json"),
            "deep_cfr_lounge_last_strategy.json"
        );
        assert_eq!(infer_strat_path("custom.json"), "custom_strategy.json");
        let bad = to_args(&["train", "--bogus-flag"]);
        assert!(parse_train_args(&bad).is_err());
    }
}

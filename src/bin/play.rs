use poker_tournament::deep_cfr::DeepCFRPolicy;
use poker_tournament::environment::FEATURE_DIM;
use poker_tournament::equity::preflop_table;
use poker_tournament::live::{
    DecisionMode, HandState, InPageAction, PlayConfig, PlayMode, process_game_update,
    set_target_game,
};
use poker_tournament::neural_network::MLP;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug)]
struct PlayCliArgs {
    config: PlayConfig,
    game_url: String,
    model_path: Option<String>,
    strategy_path: Option<String>,
    port: u16,
}

fn print_help() {
    println!("Antigravity Poker GTO Live Assistant (Tournament ICM & Lounge Chip-EV)");
    println!("Sử dụng: cargo run --release --bin play -- [OPTIONS]\n");
    println!("CHẾ ĐỘ CHƠI:");
    println!("  --mode <tournament|chip-ev|lounge>  Chọn chế độ (mặc định: tournament)");
    println!("  --tournament, --icm                 Tournament ICM (bộ model deep_cfr_*.json)");
    println!("  --lounge, --cash, --chip-ev         Lounge Cash Chip-EV (deep_cfr_lounge_*.json)");
    println!("\nMODEL & KẾT NỐI:");
    println!("  --model <path>, --adv <path>        Đường dẫn file advantage model (.json)");
    println!("  --strategy <path>, --strat <path>   Đường dẫn file strategy model (.json)");
    println!("  -g, --game-url <url>                URL cửa sổ trình duyệt để focus OS Enigo");
    println!("  --port <N>                          Cổng HTTP nhận gói Tampermonkey (mặc định: 3000)");
    println!("\nSUBGAME SOLVER & CHIẾN THUẬT:");
    println!("  --no-solver, --fast                 Tắt Subgame Solver (chỉ dùng Neural Net)");
    println!("  --search                            Luôn bật Subgame Solver ở mọi spot");
    println!("  --depth <N>                         Độ sâu tìm kiếm cây Subgame");
    println!("  --iters <N>                         Số vòng lặp CFR của Subgame Solver");
    println!("  --min-pot <BB>                      Ngưỡng Pot tối thiểu (BB) kích hoạt Solver");
    println!("  --no-turn-river-search              Không tự search Turn/River multiway");
    println!("  --roll                              Tung xúc xắc ngẫu nhiên theo % GTO");
    println!("  --guarded                           Lọc nhánh tự sát & trộn an toàn");
    println!("\nTHAO TÁC & TRỢ LÝ:");
    println!("  --advice-only, --assist             Chỉ gợi ý quyết định, KHÔNG click/bấm nút");
    println!("  --delay-min <ms>                    Độ trễ tối thiểu giả lập người (mặc định: 1500)");
    println!("  --delay-max <ms>                    Độ trễ tối đa giả lập người (mặc định: 3000)");
    println!("  --enigo                             Bật click chuột qua OS thay vì in-page DOM");
    println!("  -h, --help                          Hiển thị hướng dẫn này");
}

fn parse_mode_value(val: &str) -> Result<PlayMode, String> {
    match val.to_ascii_lowercase().as_str() {
        "tournament" | "icm" | "sng" => Ok(PlayMode::Tournament),
        "chip-ev" | "chip_ev" | "chipev" | "lounge" | "cash" => Ok(PlayMode::Lounge),
        other => Err(format!(
            "Chế độ '{}' không hợp lệ (hỗ trợ: tournament, chip-ev, lounge)",
            other
        )),
    }
}

fn next_arg<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("Thiếu giá trị cho tham số {}", flag))
}

fn parse_play_args(args: &[String]) -> Result<Option<PlayCliArgs>, String> {
    // Pass 1: xác định chế độ để khởi tạo SubgameSolverConfig mặc định phù hợp trước khi áp override
    let mut mode = PlayMode::Tournament;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--mode" => {
                let val = next_arg(args, &mut i, "--mode")?;
                mode = parse_mode_value(val)?;
            }
            "--tournament" | "--icm" => mode = PlayMode::Tournament,
            "--lounge" | "--cash" | "--chip-ev" => mode = PlayMode::Lounge,
            _ => {}
        }
        i += 1;
    }

    let mut parsed = PlayCliArgs {
        config: PlayConfig::for_mode(mode),
        game_url: "https://www.pokernow.com".to_string(),
        model_path: None,
        strategy_path: None,
        port: 3000,
    };

    // Pass 2: áp dụng đầy đủ tham số dòng lệnh
    let mut i = 1;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--mode" => {
                let _ = next_arg(args, &mut i, "--mode")?;
            }
            "--tournament" | "--icm" | "--lounge" | "--cash" | "--chip-ev" => {}
            "--game-url" | "-g" => {
                parsed.game_url = next_arg(args, &mut i, flag)?.to_string();
            }
            "--model" | "--adv" => {
                parsed.model_path = Some(next_arg(args, &mut i, flag)?.to_string());
            }
            "--strategy" | "--strat" => {
                parsed.strategy_path = Some(next_arg(args, &mut i, flag)?.to_string());
            }
            "--port" => {
                let raw = next_arg(args, &mut i, flag)?;
                parsed.port = raw
                    .parse::<u16>()
                    .map_err(|_| format!("Port '{}' không hợp lệ", raw))?;
            }
            "--no-solver" | "--fast" => parsed.config.enable_solver = false,
            "--search" => parsed.config.force_search = true,
            "--depth" => {
                let raw = next_arg(args, &mut i, flag)?;
                parsed.config.solver_config.max_depth = raw
                    .parse::<usize>()
                    .map_err(|_| format!("Depth '{}' không hợp lệ", raw))?
                    .max(1);
            }
            "--iters" => {
                let raw = next_arg(args, &mut i, flag)?;
                parsed.config.solver_config.iterations = raw
                    .parse::<usize>()
                    .map_err(|_| format!("Iterations '{}' không hợp lệ", raw))?
                    .max(10);
            }
            "--min-pot" => {
                let raw = next_arg(args, &mut i, flag)?;
                parsed.config.solver_config.min_pot_bb_trigger = raw
                    .parse::<f32>()
                    .map_err(|_| format!("Min-pot '{}' không hợp lệ", raw))?;
            }
            "--no-turn-river-search" => parsed.config.solver_config.search_turn_river = false,
            "--roll" => parsed.config.decision_mode = DecisionMode::PureGtoRoll,
            "--guarded" => parsed.config.decision_mode = DecisionMode::GuardedMixing,
            "--advice-only" | "--assist" => parsed.config.advice_only = true,
            "--delay-min" => {
                let raw = next_arg(args, &mut i, flag)?;
                parsed.config.delay_min_ms = raw
                    .parse::<u64>()
                    .map_err(|_| format!("delay-min '{}' không hợp lệ", raw))?;
            }
            "--delay-max" => {
                let raw = next_arg(args, &mut i, flag)?;
                parsed.config.delay_max_ms = raw
                    .parse::<u64>()
                    .map_err(|_| format!("delay-max '{}' không hợp lệ", raw))?;
            }
            "--enigo" => parsed.config.use_os_enigo = true,
            unknown => {
                return Err(format!("Tham số không được nhận diện: '{}'", unknown));
            }
        }
        i += 1;
    }

    parsed.config.delay_max_ms = parsed.config.delay_max_ms.max(parsed.config.delay_min_ms);
    Ok(Some(parsed))
}

fn load_mode_policy(
    mode: PlayMode,
    adv_override: Option<&str>,
    strat_override: Option<&str>,
) -> Result<(DeepCFRPolicy, String, String), String> {
    let (default_adv, default_strat) = mode.default_model_paths();
    let adv_path = adv_override.unwrap_or(default_adv).to_string();
    let strat_path = strat_override.unwrap_or(default_strat).to_string();

    let train_hint = match mode {
        PlayMode::Tournament => {
            "cargo run --release --bin train -- --mode tournament --fresh"
        }
        PlayMode::Lounge => "cargo run --release --bin train -- --mode chip-ev --fresh",
    };

    let adv_net = MLP::load_from_file(&adv_path).map_err(|e| {
        format!(
            "❌ Không thể nạp Advantage Network [{}] tại '{}': {}\n   Hãy train model đúng mode bằng lệnh:\n   {}",
            mode.label(),
            adv_path,
            e,
            train_hint
        )
    })?;

    if adv_net.input_dim() != FEATURE_DIM {
        return Err(format!(
            "❌ Kiến trúc Advantage model '{}' không khớp (có {}, cần {}). Hãy train lại: {}",
            adv_path,
            adv_net.input_dim(),
            FEATURE_DIM,
            train_hint
        ));
    }

    let strat_net = MLP::load_from_file(&strat_path).map_err(|e| {
        format!(
            "❌ Không thể nạp Strategy Network [{}] tại '{}': {}\n   Không tự fallback chéo mode. Hãy train lại:\n   {}",
            mode.label(),
            strat_path,
            e,
            train_hint
        )
    })?;

    if strat_net.input_dim() != FEATURE_DIM {
        return Err(format!(
            "❌ Kiến trúc Strategy model '{}' không khớp (có {}, cần {}). Hãy train lại: {}",
            strat_path,
            strat_net.input_dim(),
            FEATURE_DIM,
            train_hint
        ));
    }

    let adv_arc = Arc::new(RwLock::new(adv_net));
    let policy = DeepCFRPolicy::with_strategy(adv_arc, Some(strat_net), false);
    Ok((policy, adv_path, strat_path))
}

fn read_http_request(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let header_end;

    loop {
        match stream.read(&mut chunk) {
            Ok(0) => return None,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    header_end = pos + 4;
                    break;
                }
                if buf.len() > 64 * 1024 {
                    return None;
                }
            }
            Err(_) => return None,
        }
    }

    let headers_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut content_length: usize = 0;
    for line in headers_str.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            if let Ok(len) = rest.trim().parse::<usize>() {
                content_length = len;
            }
        }
    }

    // Bảo vệ tránh allocation quá lớn
    if content_length > 4 * 1024 * 1024 {
        return None;
    }

    while buf.len() < header_end + content_length {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => break,
        }
    }

    let body_end = (header_end + content_length).min(buf.len());
    let body = String::from_utf8_lossy(&buf[header_end..body_end]).to_string();
    Some((headers_str, body))
}

fn handle_client(
    mut stream: TcpStream,
    state: &mut HandState,
    policy: &DeepCFRPolicy,
    cfg: &PlayConfig,
) {
    let Some((headers, body)) = read_http_request(&mut stream) else {
        return;
    };

    if headers.starts_with("OPTIONS ") {
        let resp = "HTTP/1.1 204 No Content\r\n\
             Access-Control-Allow-Origin: *\r\n\
             Access-Control-Allow-Methods: POST, OPTIONS\r\n\
             Access-Control-Allow-Headers: Content-Type\r\n\
             Access-Control-Allow-Private-Network: true\r\n\
             Connection: close\r\n\r\n";
        let _ = stream.write_all(resp.as_bytes());
        return;
    }

    let mut pending_action: Option<InPageAction> = None;

    if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&body) {
        if arr.len() >= 2 {
            let event_name = arr[0].as_str().unwrap_or("");
            let payload = &arr[1];

            match event_name {
                "registered" => {
                    if let Some(cp) = payload.get("currentPlayer") {
                        if let Some(id) = cp
                            .get("id")
                            .or_else(|| cp.get("networkUsername"))
                            .or_else(|| cp.get("name"))
                            .and_then(|v| v.as_str())
                        {
                            if state.hero_id != id {
                                println!(
                                    "🎯 [HERO] Đã nhận diện ID: '{}' (từ registered)",
                                    id
                                );
                                state.hero_id = id.to_string();
                            }
                        }
                    }
                    if let Some(gs) = payload.get("gameState") {
                        pending_action = process_game_update(state, gs, policy, cfg);
                    }
                }
                "gC" => {
                    pending_action = process_game_update(state, payload, policy, cfg);
                }
                _ => {}
            }
        }
    }

    let json_resp = json!({
        "status": "ok",
        "bot_action": pending_action
    });
    let body_str = json_resp.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Private-Network: true\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{}",
        body_str.len(),
        body_str
    );

    if stream.write_all(resp.as_bytes()).is_err() || stream.flush().is_err() {
        if pending_action.is_some() {
            println!("⚠️ [HTTP] Gửi lệnh tới Tampermonkey thất bại — reset cờ để cho phép retry");
            state.has_evaluated_this_decision = false;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cli = match parse_play_args(&args) {
        Ok(Some(c)) => c,
        Ok(None) => {
            print_help();
            return;
        }
        Err(msg) => {
            eprintln!("❌ {}", msg);
            eprintln!("   Dùng --help để xem danh sách tham số.");
            std::process::exit(1);
        }
    };

    let mode = cli.config.mode;
    println!("================================================================");
    println!("  🤖 POKER GTO LIVE ASSISTANT [{}]", mode.label());
    println!("================================================================");

    let (policy, adv_path, strat_path) = match load_mode_policy(
        mode,
        cli.model_path.as_deref(),
        cli.strategy_path.as_deref(),
    ) {
        Ok(res) => res,
        Err(err_msg) => {
            eprintln!("{}", err_msg);
            std::process::exit(1);
        }
    };

    set_target_game(&cli.game_url);

    print!("⏳ Đang nạp bảng xác suất Preflop (169 hands × 9 opponents)...");
    let _ = std::io::stdout().flush();
    let _ = preflop_table();
    println!(" Xong!");

    println!("  ✅ Advantage Model : {}", adv_path);
    println!("  ✅ Strategy Model  : {}", strat_path);
    println!("  🎯 Decision Mode   : {:?}", cli.config.decision_mode);
    println!(
        "  🕹️ Execution       : {}",
        if cli.config.advice_only {
            "ADVICE-ONLY (Chỉ gợi ý, không auto-click)"
        } else if cli.config.use_os_enigo {
            "OS Enigo Auto-Click"
        } else {
            "Tampermonkey In-Page DOM Auto-Click"
        }
    );
    if cli.config.enable_solver {
        let sc = &cli.config.solver_config;
        println!(
            "  🧠 Subgame Solver  : ON (depth={}, iters={}, min_pot={}BB, turn_river={}, icm={}, force={})",
            sc.max_depth, sc.iterations, sc.min_pot_bb_trigger, sc.search_turn_river, sc.use_icm, cli.config.force_search
        );
    } else {
        println!("  ⚡ Subgame Solver  : OFF (Fast Neural Net Only)");
    }

    let bind_addr = format!("127.0.0.1:{}", cli.port);
    let listener = match TcpListener::bind(&bind_addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("❌ Không thể mở cổng {}: {}", bind_addr, e);
            std::process::exit(1);
        }
    };
    println!(
        "\n🌐 Đang lắng nghe Tampermonkey tại http://{} ...",
        bind_addr
    );

    let mut state = HandState::new();
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => handle_client(stream, &mut state, &policy, &cli.config),
            Err(e) => eprintln!("⚠️ Lỗi kết nối TCP: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_play_cli_defaults_tournament() {
        let args = to_args(&["play"]);
        let parsed = parse_play_args(&args).unwrap().unwrap();
        assert_eq!(parsed.config.mode, PlayMode::Tournament);
        assert!(parsed.config.solver_config.use_icm);
        assert!(parsed.config.solver_config.search_turn_river);
        assert_eq!(parsed.config.solver_config.max_depth, 3);
        assert_eq!(parsed.config.solver_config.iterations, 60);
        assert!((parsed.config.solver_config.min_pot_bb_trigger - 12.0).abs() < 1e-5);
        assert_eq!(parsed.port, 3000);
        assert!(!parsed.config.advice_only);
    }

    #[test]
    fn test_play_cli_lounge_chip_ev_overrides() {
        let args = to_args(&[
            "play",
            "--mode",
            "chip-ev",
            "--port",
            "3005",
            "--depth",
            "3",
            "--iters",
            "200",
            "--min-pot",
            "6.5",
            "--advice-only",
            "--guarded",
            "--delay-min",
            "800",
            "--delay-max",
            "500",
        ]);
        let parsed = parse_play_args(&args).unwrap().unwrap();
        assert_eq!(parsed.config.mode, PlayMode::Lounge);
        assert!(!parsed.config.solver_config.use_icm);
        assert!(parsed.config.solver_config.search_turn_river);
        assert_eq!(parsed.config.solver_config.max_depth, 3);
        assert_eq!(parsed.config.solver_config.iterations, 200);
        assert!((parsed.config.solver_config.min_pot_bb_trigger - 6.5).abs() < 1e-5);
        assert_eq!(parsed.port, 3005);
        assert!(parsed.config.advice_only);
        assert_eq!(parsed.config.decision_mode, DecisionMode::GuardedMixing);
        // delay_max được kẹp >= delay_min
        assert_eq!(parsed.config.delay_min_ms, 800);
        assert_eq!(parsed.config.delay_max_ms, 800);
    }

    #[test]
    fn test_play_cli_unknown_flag_errors() {
        let args = to_args(&["play", "--invalid-flag"]);
        assert!(parse_play_args(&args).is_err());
    }

    #[test]
    fn test_load_mode_policy_strict_missing_file_errors() {
        let err = load_mode_policy(
            PlayMode::Lounge,
            Some("/tmp/nonexistent_lounge_model_xyz.json"),
            Some("/tmp/nonexistent_lounge_strat_xyz.json"),
        )
        .err()
        .expect("expected error for missing model file");
        assert!(err.contains("deep_cfr_lounge") || err.contains("chip-ev"));
    }
}

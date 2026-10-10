use poker_tournament::deep_cfr::DeepCFRPolicy;
use poker_tournament::environment::state::ActionRecord;
use poker_tournament::environment::{Action, Environment, PlayerState, Street};
use poker_tournament::neural_network::MLP;
use poker_tournament::poker_core::card::Card;
use poker_tournament::Policy;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, RwLock};

const MODEL_PATH: &str = "deep_cfr_model.json";

#[derive(Clone, Debug)]
struct PlayerTracker {
    id: String,
    name: String,
    stack: u32,
    current_bet: u32,
    is_folded: bool,
    is_all_in: bool,
}

impl Default for PlayerTracker {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "Player".to_string(),
            stack: 1000,
            current_bet: 0,
            is_folded: false,
            is_all_in: false,
        }
    }
}

#[derive(Debug)]
struct HandState {
    hand_id: String,
    hero_id: String,
    hero_cards: Option<[Card; 2]>,
    community_cards: Vec<Card>,
    street: Street,
    main_pot: u32,
    highest_bet: u32,
    min_raise: u32,
    bb_size: u32,
    sb_size: u32,
    dealer_id: Option<String>,
    dealer_seat: Option<usize>,
    action_history: Vec<ActionRecord>,
    players: HashMap<String, PlayerTracker>,
    player_order: Vec<String>,
    has_evaluated_this_decision: bool,
}

impl HandState {
    fn new(hero_id: String) -> Self {
        Self {
            hand_id: String::new(),
            hero_id,
            hero_cards: None,
            community_cards: Vec::new(),
            street: Street::Preflop,
            main_pot: 0,
            highest_bet: 0,
            min_raise: 20,
            bb_size: 20,
            sb_size: 10,
            dealer_id: None,
            dealer_seat: None,
            action_history: Vec::new(),
            players: HashMap::new(),
            player_order: Vec::new(),
            has_evaluated_this_decision: false,
        }
    }

    fn start_new_hand(&mut self, new_h_id: &str) {
        println!("\n============================================================");
        println!("🎲 [VÁN MỚI] Hand ID: {}", new_h_id);
        println!("============================================================");
        self.hand_id = new_h_id.to_string();
        self.hero_cards = None;
        self.community_cards.clear();
        self.main_pot = 0;
        self.highest_bet = 0;
        self.street = Street::Preflop;
        self.has_evaluated_this_decision = false;
        self.dealer_id = None;
        self.dealer_seat = None;
        self.action_history.clear();
        for p in self.players.values_mut() {
            p.current_bet = 0;
            p.is_folded = false;
            p.is_all_in = false;
        }
    }

    fn total_pot(&self) -> u32 {
        let current_bets: u32 = self.players.values().map(|p| p.current_bet).sum();
        self.main_pot + current_bets
    }
}

fn parse_cards_array(val: &Value) -> Option<[Card; 2]> {
    let arr = val.as_array()?;
    let mut cards = Vec::new();

    for item in arr {
        if let Some(s) = item.as_str() {
            if let Some(card) = Card::from_str_repr(s) {
                cards.push(card);
            }
        } else if let Some(s) = item.get("value").and_then(|v| v.as_str()) {
            if let Some(card) = Card::from_str_repr(s) {
                cards.push(card);
            }
        }
    }

    if cards.len() == 2 {
        Some([cards[0], cards[1]])
    } else {
        None
    }
}

fn main() {
    println!("============================================================");
    println!("⚡ POKERNOW REAL-TIME GTO ASSISTANT (FULL OPPONENT SYNC)");
    println!("============================================================");

    print!("📐 Nạp bảng Preflop Equity... ");
    std::io::stdout().flush().unwrap();
    let _ = poker_tournament::equity::preflop_table();
    println!("✅ Xong!");

    print!("🧠 Nạp mô hình {}... ", MODEL_PATH);
    std::io::stdout().flush().unwrap();
    let mlp = MLP::load_from_file(MODEL_PATH).expect("Lỗi nạp model");
    println!("✅ Xong!");

    let policy = DeepCFRPolicy::new(Arc::new(RwLock::new(mlp)), false);

    let listener = TcpListener::bind("127.0.0.1:3000").expect("Không thể mở cổng 3000");
    println!("🌐 Server sẵn sàng tại http://127.0.0.1:3000");
    println!("👀 Đang theo dõi và đồng bộ toàn bộ người chơi...\n");

    let mut state = HandState::new("kt7ZOMBcTX".to_string());

    for stream in listener.incoming() {
        if let Ok(mut stream) = stream {
            let mut raw_data = Vec::with_capacity(16384);
            let mut temp_buf = [0u8; 8192];

            while let Ok(n) = stream.read(&mut temp_buf) {
                if n == 0 { break; }
                raw_data.extend_from_slice(&temp_buf[..n]);

                if let Some(header_end) = raw_data.windows(4).position(|w| w == b"\r\n\r\n") {
                    let header_str = String::from_utf8_lossy(&raw_data[..header_end]);

                    if header_str.starts_with("OPTIONS") {
                        let response = "HTTP/1.1 200 OK\r\n\
                                        Access-Control-Allow-Origin: *\r\n\
                                        Access-Control-Allow-Methods: POST, OPTIONS\r\n\
                                        Access-Control-Allow-Headers: Content-Type\r\n\
                                        Connection: close\r\n\
                                        Content-Length: 0\r\n\r\n";
                        let _ = stream.write_all(response.as_bytes());
                        break;
                    }

                    let mut content_len = 0usize;
                    for line in header_str.lines() {
                        if line.to_lowercase().starts_with("content-length:") {
                            if let Some(val_str) = line.split(':').nth(1) {
                                content_len = val_str.trim().parse().unwrap_or(0);
                            }
                        }
                    }

                    if raw_data.len() >= header_end + 4 + content_len {
                        let body_slice = &raw_data[header_end + 4 .. header_end + 4 + content_len];
                        let body_str = String::from_utf8_lossy(body_slice);

                        let trimmed = body_str.trim();
                        if !trimmed.is_empty() {
                            let _ = OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open("pokernow_packets.json")
                                .and_then(|mut f| writeln!(f, "{}", trimmed));

                            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                                if let Some(arr) = val.as_array() {
                                    if arr.len() >= 2 {
                                        let event_name = arr[0].as_str().unwrap_or("");
                                        let payload = &arr[1];

                                        if event_name == "registered" {
                                            if let Some(cp) = payload.get("currentPlayer") {
                                                if let Some(id) = cp.get("id").and_then(|v| v.as_str()) {
                                                    state.hero_id = id.to_string();
                                                    let name = cp.get("networkUsername").and_then(|v| v.as_str()).unwrap_or("Hero");
                                                    println!("👤 Đã kết nối: {} | Hero ID: [{}]", name, id);
                                                }
                                            }
                                            if let Some(gs) = payload.get("gameState") {
                                                process_game_update(&mut state, gs, &policy);
                                            }
                                        }

                                        if event_name == "gC" {
                                            process_game_update(&mut state, payload, &policy);
                                        }
                                    }
                                }
                            }
                        }

                        let response = "HTTP/1.1 200 OK\r\n\
                                        Access-Control-Allow-Origin: *\r\n\
                                        Connection: close\r\n\
                                        Content-Length: 2\r\n\r\nOK";
                        let _ = stream.write_all(response.as_bytes());
                        break;
                    }
                }
            }
        }
    }
}

fn process_game_update(state: &mut HandState, payload: &Value, policy: &DeepCFRPolicy) {
    if let Some(hi) = payload.get("hI").and_then(|v| v.as_str()) {
        if state.hand_id != hi {
            state.start_new_hand(hi);
        }
    }

    if let Some(bb) = payload.get("bigBlind").and_then(|v| v.as_u64()) {
        state.bb_size = bb as u32;
    }
    if let Some(sb) = payload.get("smallBlind").and_then(|v| v.as_u64()) {
        state.sb_size = sb as u32;
    }
    if let Some(p) = payload.get("pot").and_then(|v| v.as_u64()) {
        state.main_pot = p as u32;
    }
    if let Some(chb) = payload.get("cHB").and_then(|v| v.as_u64()) {
        state.highest_bet = chb as u32;
    }
    if let Some(mr) = payload.get("mR").and_then(|v| v.as_u64()) {
        state.min_raise = mr as u32;
    }

    // Danh sách người chơi trong ván theo thứ tự ngồi (iHPI)
    if let Some(ihpi) = payload.get("iHPI").and_then(|v| v.as_array()) {
        let active_ids: Vec<String> = ihpi
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        if !active_ids.is_empty() {
            state.player_order = active_ids;
        }
    }

    // Cập nhật thông tin chi tiết từng người chơi (Stack, Name)
    if let Some(players_obj) = payload.get("players").and_then(|v| v.as_object()) {
        for (pid, pval) in players_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                id: pid.clone(),
                ..Default::default()
            });
            if let Some(s) = pval.get("stack").and_then(|v| v.as_u64()) {
                entry.stack = s as u32;
            }
            if let Some(n) = pval.get("name").and_then(|v| v.as_str()) {
                entry.name = n.to_string();
            }
        }
    }

    // Cập nhật vị trí Button (Dealer)
    if let Some(did) = payload.get("dealerID").and_then(|v| v.as_str()) {
        state.dealer_id = Some(did.to_string());
    }
    if let Some(dseat) = payload.get("dealerSeat").and_then(|v| v.as_u64()) {
        state.dealer_seat = Some(dseat as usize);
    }

    // Cập nhật mức cược hiện tại của từng người chơi (tB)
    if let Some(tb_obj) = payload.get("tB").and_then(|v| v.as_object()) {
        for (pid, bval) in tb_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                id: pid.clone(),
                ..Default::default()
            });
            let prev_bet = entry.current_bet;
            if let Some(amt) = bval.as_u64() {
                let new_bet = amt as u32;
                if new_bet > prev_bet && new_bet > state.bb_size {
                    // Xác định player_idx trong player_order để tạo ActionRecord
                    let p_idx = state.player_order.iter().position(|id| id == pid).unwrap_or(0);
                    state.action_history.push(ActionRecord {
                        player_idx: p_idx,
                        street: state.street,
                        action: Action::BetPot100, // Đại diện cho hành động raise/bet được tính trong encode_features
                        amount: new_bet,
                    });
                }
                entry.current_bet = new_bet;
            } else if bval.as_str() == Some("<D>") {
                entry.current_bet = 0;
            }
        }
        if tb_obj.values().any(|v| v.as_str() == Some("<D>")) {
            state.highest_bet = 0;
        }
    }

    // CẬP NHẬT TRẠNG THÁI FOLD / ALL-IN CỦA TỪNG NGƯỜI CHƠI (pGS)
    if let Some(pgs_obj) = payload.get("pGS").and_then(|v| v.as_object()) {
        for (pid, sval) in pgs_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                id: pid.clone(),
                ..Default::default()
            });
            if let Some(st) = sval.as_str() {
                if st == "fold" {
                    entry.is_folded = true;
                } else if st == "allIn" {
                    entry.is_all_in = true;
                }
            }
        }
    }

    // Bài chung (oTC)
    if let Some(otc) = payload.get("oTC").and_then(|v| v.as_object()) {
        for (_k, c_arr) in otc {
            if let Some(arr) = c_arr.as_array() {
                let parsed_board: Vec<Card> = arr
                    .iter()
                    .filter_map(|v| v.as_str().and_then(Card::from_str_repr))
                    .collect();
                if parsed_board.len() > state.community_cards.len() {
                    let board_str: Vec<String> = parsed_board.iter().map(|c| format!("{}", c)).collect();
                    println!("🎴 [BÀI CHUNG]: [ {} ] | Pot hiện tại: {} chip", board_str.join(" "), state.total_pot());
                    state.community_cards = parsed_board;
                }
            }
        }
    }

    // Street từ gT
    if let Some(gt_arr) = payload.get("gT").and_then(|v| v.as_array()) {
        if gt_arr.len() >= 2 {
            if let Some(st_num) = gt_arr[1].as_u64() {
                state.street = match st_num {
                    0 => Street::Preflop,
                    1 => Street::Flop,
                    2 => Street::Turn,
                    3 => Street::River,
                    _ => Street::Showdown,
                };
            }
        }
    }

    // Bắt bài tẩy của Hero
    if let Some(pc_obj) = payload.get("pC").and_then(|v| v.as_object()) {
        for (pid, cdata) in pc_obj {
            let is_showdown = cdata.get("cards").and_then(|c| c.as_array())
                .map_or(false, |arr| arr.iter().any(|item| item.get("showing") == Some(&Value::Bool(true))));
            
            if is_showdown {
                continue;
            }

            let cards_val = cdata.get("cards").or_else(|| cdata.get("c"));
            if let Some(c_val) = cards_val {
                if let Some(parsed) = parse_cards_array(c_val) {
                    if state.hero_cards != Some(parsed) {
                        state.hero_id = pid.clone();
                        state.hero_cards = Some(parsed);
                        println!("🎴 [BÀI TẨY CỦA BẠN]: [ {} {} ] (Hero ID: {})", parsed[0], parsed[1], pid);
                    }
                    break;
                }
            }
        }
    }

    // Lượt hành động
    let current_actor = payload.get("pITT").or_else(|| payload.get("cPI")).and_then(|v| v.as_str());

    if let Some(actor) = current_actor {
        if let Some(mavtb) = payload.get("mAVTB").and_then(|v| v.as_u64()) {
            if let Some(p) = state.players.get_mut(actor) {
                p.stack = mavtb as u32;
            }
        }

        if actor == state.hero_id {
            if !state.has_evaluated_this_decision {
                if let Some(hole) = state.hero_cards {
                    state.has_evaluated_this_decision = true;
                    run_gto_evaluation(state, hole, policy);
                } else {
                    println!("⏳ Đến lượt bạn nhưng ván này chưa nhận diện được bài tẩy!");
                }
            }
        } else {
            state.has_evaluated_this_decision = false;
        }
    }
}

/// ĐỒNG BỘ 100% ĐỐI THỦ VÀO MÔI TRƯỜNG & CHẠY GTO
fn run_gto_evaluation(state: &HandState, hole: [Card; 2], policy: &DeepCFRPolicy) {
    let seat_ids: Vec<String> = if !state.player_order.is_empty() {
        state.player_order.clone()
    } else {
        state.players.keys().cloned().collect()
    };

    let num_players = seat_ids.len().clamp(2, 8);
    let hero_seat = seat_ids.iter().position(|id| id == &state.hero_id).unwrap_or(0).min(num_players - 1);

    // 1. Tạo danh sách PlayerState thật cho toàn bộ bàn đấu
    let mut initial_stacks = Vec::with_capacity(num_players);
    let mut env_players = Vec::with_capacity(num_players);

    for (seat_idx, pid) in seat_ids.iter().take(num_players).enumerate() {
        let p_track = state.players.get(pid).cloned().unwrap_or_default();
        let stack = p_track.stack.max(state.bb_size * 2);
        initial_stacks.push(stack);

        let mut ps = PlayerState::new(seat_idx, stack, seat_idx != hero_seat);
        ps.current_bet = p_track.current_bet;
        ps.is_folded = p_track.is_folded;
        ps.is_all_in = p_track.is_all_in;
        if seat_idx == hero_seat {
            ps.hole_cards = [Some(hole[0]), Some(hole[1])];
        }
        env_players.push(ps);
    }

    // Xác định vị trí Dealer Button trong số các ghế hiện tại
    let button_idx = if let Some(ref did) = state.dealer_id {
        seat_ids.iter().position(|id| id == did).unwrap_or(0)
    } else if let Some(dseat) = state.dealer_seat {
        dseat.saturating_sub(1).min(num_players - 1)
    } else {
        0
    };

    let mut env = Environment::new(num_players, &initial_stacks, state.bb_size, state.sb_size, button_idx);

    // 2. Đồng bộ toàn bộ trạng thái thực tế vào env
    let total_pot = state.total_pot().max(state.bb_size + state.sb_size);
    env.state.pot = total_pot;
    env.state.current_highest_bet = state.highest_bet;
    env.state.community_cards = state.community_cards.clone();
    env.state.street = state.street;
    env.state.current_player_idx = hero_seat;
    env.state.button_idx = button_idx;
    env.state.action_history = state.action_history.clone();
    env.state.players = env_players; // <-- NẠP TOÀN BỘ ĐỐI THỦ THẬT VÀO ĐÂY!

    let hero_bet = state.players.get(&state.hero_id).map(|p| p.current_bet).unwrap_or(0);
    let hero_stack = state.players.get(&state.hero_id).map(|p| p.stack).unwrap_or(1000);
    let to_call = state.highest_bet.saturating_sub(hero_bet);

    // 3. Tính toán GTO
    let mask = env.get_action_mask();
    let features = env.state.encode_features(hero_seat);
    let probs = policy.get_action_probs(&features, &mask);

    println!("\n============================================================");
    println!("🎯 [ĐẾN LƯỢT BẠN HÀNH ĐỘNG] - VÒNG: {:?}", state.street);
    println!("🃏 Bài tẩy : [ {} {} ]", hole[0], hole[1]);
    if !state.community_cards.is_empty() {
        let board_str: Vec<String> = state.community_cards.iter().map(|c| format!("{}", c)).collect();
        println!("🎴 Bài chung: [ {} ]", board_str.join(" "));
    }
    println!("💰 Tổng Pot: {} chip | Cần Call: {} chip | Stack: {} chip", total_pot, to_call, hero_stack);

    // In danh sách trực quan tình trạng các đối thủ trên bàn
    println!("------------------------------------------------------------");
    println!("👥 TRẠNG THÁI CÁC GHẾ TRÊN BÀN:");
    for (idx, pid) in seat_ids.iter().take(num_players).enumerate() {
        let p = state.players.get(pid).cloned().unwrap_or_default();
        let role = if pid == &state.hero_id { "⭐ HERO (Bạn)" } else { "Đối thủ" };
        let status = if p.is_folded {
            "🛑 Đã Fold"
        } else if p.is_all_in {
            "🚀 ALL-IN"
        } else if p.current_bet > 0 {
            "🪙 Đã cược"
        } else {
            "🟢 Đang theo"
        };
        println!("   [Ghế {}] {:<14} | {:<12} | Stack: {:>5} | Bet: {:>4} | {}", 
            idx + 1, p.name, role, p.stack, p.current_bet, status);
    }

    println!("------------------------------------------------------------");
    println!("📊 TỶ LỆ HÀNH ĐỘNG GTO (DEEP CFR):");

    let mut best_idx = 0;
    let mut best_p = -1.0f32;
    for (i, &p) in probs.iter().enumerate() {
        if mask.is_index_valid(i) && p > best_p {
            best_p = p;
            best_idx = i;
        }
    }

    for (i, &p) in probs.iter().enumerate() {
        let act = Action::from_index(i).unwrap();
        let is_valid = mask.is_index_valid(i);
        let rec = if i == best_idx { " ⭐ KHUYÊN DÙNG" } else { "" };
        let status = if is_valid { "" } else { " (Không hợp lệ)" };

        println!("   {:<14}: {:>5.1}%{}{}", act.name(), p * 100.0, rec, status);
    }
    println!("============================================================\n");
}

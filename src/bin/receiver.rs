use poker_tournament::deep_cfr::DeepCFRPolicy;
use poker_tournament::environment::state::ActionRecord;
use poker_tournament::environment::{Action, Environment, PlayerState, Street};
use poker_tournament::neural_network::MLP;
use poker_tournament::poker_core::{Card, Rank, Suit};
use poker_tournament::Policy;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, RwLock};

const ADV_MODEL_PATH: &str = "deep_cfr_model.json";
const STRAT_MODEL_PATH: &str = "deep_cfr_strategy.json";

#[derive(Clone, Debug)]
struct PlayerTracker {
    id: String,
    name: String,
    seat_num: usize,
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
            seat_num: 99,
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
    pn_total_pot: u32,
    highest_bet: u32,
    min_raise: u32,
    bb_size: u32,
    sb_size: u32,
    dealer_id: Option<String>,
    dealer_seat: Option<usize>,
    action_history: Vec<ActionRecord>,
    players: HashMap<String, PlayerTracker>,
    seats_order: Vec<String>,
    has_evaluated_this_decision: bool,
}

impl HandState {
    fn new() -> Self {
        Self {
            hand_id: String::new(),
            hero_id: String::new(),
            hero_cards: None,
            community_cards: Vec::new(),
            street: Street::Preflop,
            pn_total_pot: 0,
            highest_bet: 0,
            min_raise: 20,
            bb_size: 20,
            sb_size: 10,
            dealer_id: None,
            dealer_seat: None,
            action_history: Vec::new(),
            players: HashMap::new(),
            seats_order: Vec::new(),
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
        self.pn_total_pot = 0;
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
}

fn robust_parse_card(s: &str) -> Option<Card> {
    let s = s.trim();
    if s.len() < 2 || s.len() > 3 { return None; }
    let (r_char, s_char) = if s.len() == 3 && s.starts_with("10") {
        ('T', s.chars().nth(2)?)
    } else {
        let mut it = s.chars();
        (it.next()?, it.next()?)
    };

    let rank = match r_char {
        '2' => Rank::Two, '3' => Rank::Three, '4' => Rank::Four, '5' => Rank::Five,
        '6' => Rank::Six, '7' => Rank::Seven, '8' => Rank::Eight, '9' => Rank::Nine,
        'T' | 't' => Rank::Ten, 'J' | 'j' => Rank::Jack, 'Q' | 'q' => Rank::Queen,
        'K' | 'k' => Rank::King, 'A' | 'a' => Rank::Ace, _ => return None,
    };
    let suit = match s_char {
        's' | 'S' => Suit::Spades, 'h' | 'H' => Suit::Hearts,
        'd' | 'D' => Suit::Diamonds, 'c' | 'C' => Suit::Clubs, _ => return None,
    };
    Some(Card::new(rank, suit))
}

fn parse_cards_array(val: &Value) -> Option<[Card; 2]> {
    let arr = val.as_array()?;
    let mut cards = Vec::new();
    for item in arr {
        if let Some(s) = item.as_str() { if let Some(card) = robust_parse_card(s) { cards.push(card); } }
        else if let Some(s) = item.get("value").and_then(|v| v.as_str()) { if let Some(card) = robust_parse_card(s) { cards.push(card); } }
    }
    if cards.len() == 2 { Some([cards[0], cards[1]]) } else { None }
}

fn main() {
    println!("============================================================");
    println!("⚡ POKERNOW REAL-TIME GTO ASSISTANT (HIGH PRECISION v3)");
    println!("============================================================");

    print!("📐 Nạp bảng Preflop Equity... ");
    std::io::stdout().flush().unwrap();
    let _ = poker_tournament::equity::preflop_table();
    println!("✅ Xong!");

    print!("🧠 Nạp mô hình AI... ");
    std::io::stdout().flush().unwrap();
    let adv_mlp = MLP::load_from_file(ADV_MODEL_PATH).expect("Lỗi nạp deep_cfr_model.json");
    let strat_mlp = MLP::load_from_file(STRAT_MODEL_PATH).ok();
    println!("✅ Xong! (Strategy Net: {})", if strat_mlp.is_some() { "Có" } else { "Không" });

    let policy = DeepCFRPolicy::with_strategy(Arc::new(RwLock::new(adv_mlp)), strat_mlp, false);

    let listener = TcpListener::bind("127.0.0.1:3000").expect("Không thể mở cổng 3000");
    println!("🌐 Server sẵn sàng tại http://127.0.0.1:3000");
    println!("👀 Đang theo dõi và đồng bộ PokerNow...\n");

    let mut state = HandState::new();

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
                        let response = "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
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
                            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                                if let Some(arr) = val.as_array() {
                                    if arr.len() >= 2 {
                                        let event_name = arr[0].as_str().unwrap_or("");
                                        let payload = &arr[1];

                                        if event_name == "registered" {
                                            if let Some(cp) = payload.get("currentPlayer") {
                                                if let Some(id) = cp.get("id").and_then(|v| v.as_str()) {
                                                    state.hero_id = id.to_string();
                                                    let name = cp.get("networkUsername").or_else(|| cp.get("name")).and_then(|v| v.as_str()).unwrap_or("Hero");
                                                    println!("👤 Đã kết nối: {} | Hero ID: [{}]", name, id);
                                                }
                                            }
                                            if let Some(gs) = payload.get("gameState") {
                                                process_game_update(&mut state, gs, &policy);
                                            }
                                        } else if event_name == "gC" {
                                            process_game_update(&mut state, payload, &policy);
                                        }
                                    }
                                }
                            }
                        }

                        let response = "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\nContent-Length: 2\r\n\r\nOK";
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
        if state.hand_id != hi { state.start_new_hand(hi); }
    }

    if let Some(gt_arr) = payload.get("gT").and_then(|v| v.as_array()) {
        if gt_arr.len() >= 2 {
            if let Some(st_num) = gt_arr[1].as_u64() {
                let new_street = match st_num {
                    0 => Street::Preflop, 1 => Street::Flop, 2 => Street::Turn,
                    3 => Street::River, _ => Street::Showdown,
                };
                if new_street != state.street {
                    for p in state.players.values_mut() {
                        p.stack = p.stack.saturating_sub(p.current_bet);
                        p.current_bet = 0;
                    }
                    state.highest_bet = 0;
                    state.street = new_street;
                    state.has_evaluated_this_decision = false;
                }
            }
        }
    }

    if let Some(arr) = payload.get("seats").and_then(|v| v.as_array()) {
        let mut v: Vec<(u64, String)> = arr.iter().filter_map(|s| {
            let a = s.as_array()?;
            Some((a.get(0)?.as_u64()?, a.get(1)?.as_str()?.to_string()))
        }).collect();
        v.sort();
        state.seats_order = v.into_iter().map(|(_, id)| id).collect();
    }

    if let Some(bb) = payload.get("bigBlind").and_then(|v| v.as_u64()) { state.bb_size = bb as u32; }
    if let Some(sb) = payload.get("smallBlind").and_then(|v| v.as_u64()) { state.sb_size = sb as u32; }
    
    if let Some(p) = payload.get("pot") {
        if let Some(num) = p.as_u64() {
            state.pn_total_pot = num as u32;
        } else if let Some(arr) = p.as_array() {
            state.pn_total_pot = arr.iter().filter_map(|v| v.as_u64()).sum::<u64>() as u32;
        }
    }

    if let Some(chb) = payload.get("cHB").and_then(|v| v.as_u64()) { state.highest_bet = chb as u32; }
    
    if let Some(mr) = payload.get("mR").and_then(|v| v.as_u64()) {
        state.min_raise = (mr as u32).saturating_sub(state.highest_bet).max(state.bb_size);
    }

    if let Some(players_obj) = payload.get("players").and_then(|v| v.as_object()) {
        for (pid, pval) in players_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                id: pid.clone(), ..Default::default()
            });
            if let Some(s) = pval.get("stack").and_then(|v| v.as_u64()) { entry.stack = s as u32; }
            if let Some(n) = pval.get("name").and_then(|v| v.as_str()) { entry.name = n.to_string(); }
            if let Some(seat) = pval.get("seat").and_then(|v| v.as_u64()) { entry.seat_num = seat as usize; }
        }
    }

    if let Some(did) = payload.get("dealerID").and_then(|v| v.as_str()) {
        state.dealer_id = Some(did.to_string());
    }

    if let Some(tb_obj) = payload.get("tB").and_then(|v| v.as_object()) {
        for (pid, bval) in tb_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker { id: pid.clone(), ..Default::default() });
            let prev_bet = entry.current_bet;
            if let Some(amt) = bval.as_u64() {
                let new_bet = amt as u32;
                if new_bet > prev_bet && new_bet > state.bb_size {
                    if pid != &state.hero_id { state.has_evaluated_this_decision = false; }
                }
                entry.current_bet = new_bet;
            } else if bval.as_str() == Some("<D>") {
                entry.current_bet = 0;
            }
        }
    }

    if let Some(pgs_obj) = payload.get("pGS").and_then(|v| v.as_object()) {
        for (pid, sval) in pgs_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker { id: pid.clone(), ..Default::default() });
            if let Some(st) = sval.as_str() {
                if st == "fold" { entry.is_folded = true; } 
                else if st == "allIn" { entry.is_all_in = true; }
            }
        }
    }

    if let Some(ihpi) = payload.get("iHPI").and_then(|v| v.as_array()) {
        let active_set: std::collections::HashSet<String> = ihpi.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
        if !active_set.is_empty() {
            for (pid, p) in state.players.iter_mut() {
                if !active_set.contains(pid) { p.is_folded = true; }
            }
        }
    }

    if let Some(otc) = payload.get("oTC").and_then(|v| v.as_object()) {
        let mut new_board = Vec::new();
        for (_k, c_arr) in otc {
            if let Some(arr) = c_arr.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str() { if let Some(c) = robust_parse_card(s) { new_board.push(c); } }
                }
            }
        }
        if new_board.len() > state.community_cards.len() {
            let board_str: Vec<String> = new_board.iter().map(|c| format!("{}", c)).collect();
            println!("🎴 [BÀI CHUNG]: [ {} ]", board_str.join(" "));
            state.community_cards = new_board;
            state.has_evaluated_this_decision = false;
        }
    }

    if let Some(pc_obj) = payload.get("pC").and_then(|v| v.as_object()) {
        for (pid, cdata) in pc_obj {
            let is_showdown = cdata.get("cards").and_then(|c| c.as_array()).map_or(false, |arr| arr.iter().any(|item| item.get("showing") == Some(&Value::Bool(true))));
            if is_showdown { continue; }

            let cards_val = cdata.get("cards").or_else(|| cdata.get("c"));
            if let Some(c_val) = cards_val {
                if let Some(parsed) = parse_cards_array(c_val) {
                    if state.hero_id.is_empty() || pid == &state.hero_id {
                        state.hero_id = pid.clone();
                        if state.hero_cards != Some(parsed) {
                            state.hero_cards = Some(parsed);
                            println!("🎴 [BÀI TẨY CỦA BẠN]: [ {} {} ] (Hero ID: {})", parsed[0], parsed[1], pid);
                        }
                    }
                }
            }
        }
    }

    let current_actor = payload.get("pITT").or_else(|| payload.get("cPI")).and_then(|v| v.as_str());

    if let Some(actor) = current_actor {
        if let Some(mavtb) = payload.get("mAVTB").and_then(|v| v.as_u64()) {
            if let Some(p) = state.players.get_mut(actor) { p.stack = mavtb as u32; }
        }

        if !state.hero_id.is_empty() && actor == state.hero_id {
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

fn run_gto_evaluation(state: &HandState, hole: [Card; 2], policy: &DeepCFRPolicy) {
    let mut sorted_players: Vec<&PlayerTracker> = state.players.values().collect();
    sorted_players.sort_by_key(|p| p.seat_num);

    let active_seats: Vec<String> = sorted_players
        .into_iter()
        .filter(|p| p.id == state.hero_id || p.stack > 0 || p.current_bet > 0 || p.is_all_in)
        .map(|p| p.id.clone())
        .collect();

    if active_seats.is_empty() { return; }

    let num_players = active_seats.len().clamp(2, 8);
    let hero_seat = active_seats.iter().position(|id| id == &state.hero_id).unwrap_or(0);

    // ====================================================================
    // 1. QUÉT DỮ LIỆU THỰC TẾ TRÊN BÀN (CHIP CƯỢC & POT)
    // ====================================================================
    let mut actual_highest_bet = state.highest_bet;
    let mut sum_current_bets = 0;

    let mut initial_stacks = Vec::with_capacity(num_players);
    let mut env_players = Vec::with_capacity(num_players);

    for (seat_idx, pid) in active_seats.iter().enumerate() {
        let p_track = state.players.get(pid).cloned().unwrap_or_default();
        
        // Quét tìm mức bet cao nhất hiện tại
        if p_track.current_bet > actual_highest_bet {
            actual_highest_bet = p_track.current_bet;
        }
        
        let true_stack_behind = p_track.stack.saturating_sub(p_track.current_bet);
        let starting_stack = true_stack_behind + p_track.current_bet;
        initial_stacks.push(starting_stack.max(state.bb_size));

        let mut ps = PlayerState::new(seat_idx, true_stack_behind, seat_idx != hero_seat);
        ps.current_bet = p_track.current_bet;
        ps.is_folded = p_track.is_folded;
        ps.is_all_in = p_track.is_all_in;
        if seat_idx == hero_seat { ps.hole_cards = [Some(hole[0]), Some(hole[1])]; }
        
        sum_current_bets += p_track.current_bet;
        env_players.push(ps);
    }

    let button_idx = if let Some(ref did) = state.dealer_id { active_seats.iter().position(|id| id == did).unwrap_or(0) } else { 0 };
    let mut env = Environment::new(num_players, &initial_stacks, state.bb_size, state.sb_size, button_idx);

    // ====================================================================
    // 2. TÍNH POT CHUẨN XÁC (KHÔNG BỊ ĐẾM TRÙNG, KHÔNG BỊ THIẾU)
    // ====================================================================
    // Tổng số chip thực sự đang có trên bàn = Main Pot (vòng trước) + Toàn bộ tiền cược vòng này
    let true_table_pot = (state.pn_total_pot + sum_current_bets).max(state.bb_size + state.sb_size);
    let safe_min_raise = state.min_raise.max(state.bb_size);
    
    env.state.pot = true_table_pot;
    env.state.current_highest_bet = actual_highest_bet;
    env.state.min_raise = safe_min_raise;
    env.state.community_cards = state.community_cards.clone();
    env.state.street = state.street;
    env.state.current_player_idx = hero_seat;
    env.state.button_idx = button_idx;
    env.state.players = env_players;

    let hero_bet = state.players.get(&state.hero_id).map(|p| p.current_bet).unwrap_or(0);
    let hero_stack_behind = env.state.players[hero_seat].stack; 
    let to_call = actual_highest_bet.saturating_sub(hero_bet);

    // ====================================================================
    // 3. SUY LUẬN AI (DEEP CFR GTO)
    // ====================================================================
    let mask = env.get_action_mask();
    let features = env.state.encode_features(hero_seat);
    let mut probs = policy.get_action_probs(&features, &mask);
    let raw_adv = policy.advantage_net.read().unwrap().forward(&features);

    let num_acts = raw_adv.len().min(probs.len());
    if policy.strategy_net.read().unwrap().is_none() {
        let mut max_val = f32::NEG_INFINITY;
        for (a_idx, &adv) in raw_adv.iter().enumerate().take(num_acts) {
            if mask.is_index_valid(a_idx) && adv > max_val { max_val = adv; }
        }
        let mut sum_exp = 0.0f32;
        let temp = 0.15;
        for (a_idx, &adv) in raw_adv.iter().enumerate().take(num_acts) {
            if mask.is_index_valid(a_idx) {
                let e = ((adv - max_val) / temp).exp();
                probs[a_idx] = e; sum_exp += e;
            } else { probs[a_idx] = 0.0; }
        }
        if sum_exp > 0.0 { for p in &mut probs { *p /= sum_exp; } }
    }

    // ====================================================================
    // 4. HIỂN THỊ THÔNG TIN VÁN BÀI
    // ====================================================================
    println!("\n============================================================");
    println!("🎯 [ĐẾN LƯỢT BẠN] - VÒNG: {:?}", state.street);
    println!("🃏 Bài tẩy : [ {} {} ]", hole[0], hole[1]);
    if !state.community_cards.is_empty() {
        let board_str: Vec<String> = state.community_cards.iter().map(|c| format!("{}", c)).collect();
        println!("🎴 Bài chung: [ {} ]", board_str.join(" "));
    }
    
    let call_status = if to_call == 0 { "(Đang check)" } else { "" };
    println!("💰 Tổng Pot thực tế: {} chip | Cần Call: {} chip {}", true_table_pot, to_call, call_status);
    println!("------------------------------------------------------------");

    let mut best_idx = 0;
    let mut best_p = -1.0f32;
    for (i, &p) in probs.iter().enumerate() {
        if mask.is_index_valid(i) && p > best_p {
            best_p = p; best_idx = i;
        }
    }

    // ====================================================================
    // 5. TÍNH TOÁN CON SỐ CHIP TỐ THEO CHUẨN POT-LIMIT (KHỚP 100% POKERNOW)
    // ====================================================================
    let get_action_desc = |act: Action| -> String {
        match act {
            Action::Fold => "0".to_string(),
            Action::CheckCall => {
                if to_call == 0 { "Check".to_string() } 
                else if hero_stack_behind <= to_call { format!("Call All-in ({})", hero_stack_behind) } 
                else { format!("Call {}", to_call) }
            },
            Action::AllIn => format!("All-In ({})", hero_stack_behind),
            Action::BetPot33 | Action::BetPot66 | Action::BetPot100 => {
                let frac = match act { 
                    Action::BetPot33 => 0.33, 
                    Action::BetPot66 => 0.66, 
                    _ => 1.0 
                };
                
                // Công thức chuẩn quốc tế:
                // Pot sau khi Call = Pot hiện tại trên bàn + Số tiền mình phải bỏ ra để Call
                let pot_after_call = true_table_pot + to_call;
                let raise_by = ((pot_after_call as f32) * frac).round() as u32;
                
                // Tổng số cược đẩy ra = Mức cược đối thủ + Tiền tố thêm (Tối thiểu bằng Min Raise)
                let target_total = actual_highest_bet + raise_by.max(safe_min_raise);
                format!("Tố lên {}", target_total)
            }
        }
    };

    println!("📊 KHUYẾN NGHỊ TỪ GTO SOLVER:");
    for (i, &p) in probs.iter().enumerate() {
        let act = Action::from_index(i).unwrap();
        let is_valid = mask.is_index_valid(i);
        let rec = if is_valid && i == best_idx { " ⭐ TỐT NHẤT" } else { "" };
        
        let status = if is_valid { 
            "" 
        } else { 
            let hero_stack_bb = hero_stack_behind as f32 / state.bb_size as f32;
            if hero_stack_bb < 12.0 && act != Action::Fold && act != Action::CheckCall && act != Action::AllIn {
                " (Khóa do Stack < 12BB - Push/Fold Mode)"
            } else if act == Action::BetPot66 || act == Action::BetPot100 {
                " (Khóa do trùng size/hết chip)"
            } else {
                " (Bị khóa)"
            }
        };

        let raw_val = raw_adv.get(i).copied().unwrap_or(0.0);
        let chip_desc = get_action_desc(act);

        println!("   {:<14}: {:>5.1}% (EV: {:>+6.2}) | [{:<15}]{}{}", act.name(), p * 100.0, raw_val, chip_desc, rec, status);
    }
    println!("============================================================\n");
}

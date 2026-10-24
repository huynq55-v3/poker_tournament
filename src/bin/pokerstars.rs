use poker_tournament::deep_cfr::DeepCFRPolicy;
use poker_tournament::environment::state::ActionRecord;
use poker_tournament::environment::{Action, Environment, NUM_ACTIONS, PlayerState, Street};
use poker_tournament::neural_network::MLP;
use poker_tournament::poker_core::{Card, Rank, Suit};
use poker_tournament::Policy;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, RwLock};
use std::{thread, time::Duration};
use rand::Rng;

use std::sync::atomic::{AtomicBool, Ordering};
use enigo::{Enigo, Key, Keyboard, Settings};

static IS_CASH_GAME: AtomicBool = AtomicBool::new(true);

const CASH_ADV_MODEL_PATH: &str = "deep_cfr_cash_model.json";
const CASH_STRAT_MODEL_PATH: &str = "deep_cfr_cash_strategy.json";
const TOUR_ADV_MODEL_PATH: &str = "deep_cfr_model_tour.json";
const TOUR_STRAT_MODEL_PATH: &str = "deep_cfr_strategy_tour.json";

fn get_timestamp_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

// =========================================================================
// 1. DATA STRUCTURES & TYPES
// =========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecisionMode {
    PureMaxEv,     // Mặc định: 100% Argmax EV (Thuần EV Max)
    GuardedMixing, // --guarded: Lọc phòng hộ EV + Bảo vệ range GTO
    PureGtoRoll,   // --roll: 100% GTO Roll nguyên bản
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct InPageAction {
    pub action: String, // "fold", "check", "call", "all_in", "raise"
    pub amount: u32,
    pub to_call: u32,
    #[serde(default)]
    pub delay_ms: u64,
}

#[derive(Clone, Debug)]
pub struct HandActionRecord {
    pub player_id: String,
    pub seat_num: usize,
    pub street: Street,
    pub action: Action,
    pub amount: u32,
}

#[derive(Clone, Debug)]
pub struct PlayerTracker {
    pub id: String,
    pub name: String,
    pub seat_num: usize,
    pub stack: u32,
    pub current_bet: u32,
    pub total_contributed: u32,
    pub is_folded: bool,
    pub is_all_in: bool,
    pub is_sitting_out: bool,
    pub is_in_hand: bool,
}

pub fn extract_user_id_from_payload(val: &Value) -> Option<String> {
    if let Some(uid) = val.get("hero_id").or_else(|| val.get("user_id")).and_then(|v| v.as_str()) {
        if !uid.is_empty() { return Some(uid.to_string()); }
    }
    let data_str = if let Some(s) = val.get("data").and_then(|v| v.as_str()) {
        s.to_string()
    } else if let Some(arr) = val.get("data").and_then(|v| v.as_array()) {
        arr.iter().map(|item| item.as_str().unwrap_or("")).collect::<Vec<_>>().join(" ")
    } else {
        val.to_string()
    };

    // 1. "userId":"username"
    if let Some(idx) = data_str.find("userId") {
        let tail = &data_str[idx + 6..];
        let cleaned = tail.trim_start_matches(|c| c == '\\' || c == '"' || c == ':' || c == ' ');
        if let Some(end) = cleaned.find(|c| c == '\\' || c == '"' || c == ',' || c == '}' || c == ' ' || c == '\n') {
            let u = cleaned[..end].trim();
            if !u.is_empty() && u.len() <= 24 {
                return Some(u.to_string());
            }
        }
    }

    // 2. notifications/.../<username>
    if let Some(idx) = data_str.find("notifications/") {
        let sub = &data_str[idx + 14..];
        let parts: Vec<&str> = sub.split(|c| c == '/' || c == '\n' || c == '\\' || c == '"').collect();
        for s in parts {
            let s_trim = s.trim();
            if s_trim.len() >= 3 && !["en", "1", "512", "0", "user", "queue", "starsweb"].contains(&s_trim) 
               && s_trim.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
                return Some(s_trim.to_string());
            }
        }
    }

    None
}

impl Default for PlayerTracker {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "Player".to_string(),
            seat_num: 99,
            stack: 0, // Đọc 100% từ socket, không hardcode
            current_bet: 0,
            total_contributed: 0,
            is_folded: false,
            is_all_in: false,
            is_sitting_out: false,
            is_in_hand: true,
        }
    }
}

#[derive(Debug)]
pub struct HandState {
    pub hand_id: String,
    pub table_id: String,
    pub table_name: String,
    pub hero_id: String,
    pub hero_seat: Option<usize>,
    pub hero_cards: Option<[Card; 2]>,
    pub community_cards: Vec<Card>,
    pub street: Street,
    pub total_pot: u32,
    pub rake: u32,
    pub highest_bet: u32,
    pub min_raise: u32,
    pub bb_size: u32,
    pub sb_size: u32,
    pub dealer_seat: Option<usize>,
    pub action_history: Vec<HandActionRecord>,
    pub players: HashMap<String, PlayerTracker>,
    pub has_evaluated_this_decision: bool,
    pub current_turn_seat: Option<usize>,
}

impl HandState {
    pub fn new() -> Self {
        Self {
            hand_id: String::new(),
            table_id: String::new(),
            table_name: String::new(),
            hero_id: String::new(), // Tự động đọc từ socket, không hardcode
            hero_seat: None,
            hero_cards: None,
            community_cards: Vec::new(),
            street: Street::Preflop,
            total_pot: 0,
            rake: 0,
            highest_bet: 0,
            min_raise: 0, // Đọc từ socket 0x006a
            bb_size: 0,   // Đọc từ socket 0x006a
            sb_size: 0,   // Đọc từ socket 0x006a
            dealer_seat: None,
            action_history: Vec::new(),
            players: HashMap::new(),
            has_evaluated_this_decision: false,
            current_turn_seat: None,
        }
    }

    pub fn start_new_hand(&mut self, new_h_id: &str) {
        println!("\n============================================================");
        println!("🎲 [VÁN MỚI POKERSTARS] Hand ID: {}", new_h_id);
        println!("============================================================");
        self.hand_id = new_h_id.to_string();
        self.hero_cards = None;
        self.community_cards.clear();
        self.total_pot = 0;
        self.rake = 0;
        self.highest_bet = 0;
        self.min_raise = self.bb_size;
        self.street = Street::Preflop;
        self.has_evaluated_this_decision = false;
        self.current_turn_seat = None;
        self.action_history.clear();

        if new_h_id.starts_with("6Z.") {
            self.players.clear();
        } else {
            for p in self.players.values_mut() {
                p.current_bet = 0;
                p.total_contributed = 0;
                p.is_folded = false;
                p.is_all_in = false;
            }
        }
    }
}

// =========================================================================
// 2. APACHE THRIFT BINARY PROTOCOL DECODER
// =========================================================================

#[derive(Debug, Clone)]
pub enum ThriftVal {
    Bool(bool),
    Byte(u8),
    I16(i16),
    I32(i32),
    I64(i64),
    Double(f64),
    Binary(Vec<u8>),
    String(String),
    Struct(HashMap<i16, ThriftVal>),
    List(Vec<ThriftVal>),
    Map(Vec<(ThriftVal, ThriftVal)>),
}

impl ThriftVal {
    pub fn get_i32(&self) -> Option<i32> {
        match self {
            ThriftVal::I32(v) => Some(*v),
            ThriftVal::I16(v) => Some(*v as i32),
            ThriftVal::Byte(v) => Some(*v as i32),
            _ => None,
        }
    }

    pub fn get_str(&self) -> Option<&str> {
        match self {
            ThriftVal::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_field(&self, fid: i16) -> Option<&ThriftVal> {
        match self {
            ThriftVal::Struct(map) => map.get(&fid),
            _ => None,
        }
    }

    pub fn get_list(&self) -> Option<&[ThriftVal]> {
        match self {
            ThriftVal::List(list) => Some(list.as_slice()),
            _ => None,
        }
    }
}

pub fn parse_thrift_val(t: u8, b: &[u8], mut offset: usize) -> Option<(ThriftVal, usize)> {
    match t {
        2 => { // BOOL
            if offset >= b.len() { return None; }
            let val = b[offset] != 0;
            Some((ThriftVal::Bool(val), offset + 1))
        }
        3 => { // BYTE
            if offset >= b.len() { return None; }
            Some((ThriftVal::Byte(b[offset]), offset + 1))
        }
        4 => { // DOUBLE
            if offset + 8 > b.len() { return None; }
            let bytes: [u8; 8] = b[offset..offset+8].try_into().ok()?;
            Some((ThriftVal::Double(f64::from_be_bytes(bytes)), offset + 8))
        }
        6 => { // I16
            if offset + 2 > b.len() { return None; }
            let bytes: [u8; 2] = b[offset..offset+2].try_into().ok()?;
            Some((ThriftVal::I16(i16::from_be_bytes(bytes)), offset + 2))
        }
        8 => { // I32
            if offset + 4 > b.len() { return None; }
            let bytes: [u8; 4] = b[offset..offset+4].try_into().ok()?;
            Some((ThriftVal::I32(i32::from_be_bytes(bytes)), offset + 4))
        }
        10 => { // I64
            if offset + 8 > b.len() { return None; }
            let bytes: [u8; 8] = b[offset..offset+8].try_into().ok()?;
            Some((ThriftVal::I64(i64::from_be_bytes(bytes)), offset + 8))
        }
        11 => { // STRING / BINARY
            if offset + 4 > b.len() { return None; }
            let len = u32::from_be_bytes(b[offset..offset+4].try_into().ok()?) as usize;
            offset += 4;
            if offset + len > b.len() { return None; }
            let slice = &b[offset..offset+len];
            let val = match std::str::from_utf8(slice) {
                Ok(s) => ThriftVal::String(s.to_string()),
                Err(_) => ThriftVal::Binary(slice.to_vec()),
            };
            Some((val, offset + len))
        }
        12 => { // STRUCT
            let mut fields = HashMap::new();
            while offset < b.len() {
                let ft = b[offset];
                offset += 1;
                if ft == 0 { break; } // STOP
                if offset + 2 > b.len() { break; }
                let fid = i16::from_be_bytes(b[offset..offset+2].try_into().ok()?);
                offset += 2;
                if let Some((v, next_off)) = parse_thrift_val(ft, b, offset) {
                    fields.insert(fid, v);
                    offset = next_off;
                } else {
                    break;
                }
            }
            Some((ThriftVal::Struct(fields), offset))
        }
        14 | 15 => { // SET or LIST
            if offset + 5 > b.len() { return None; }
            let elem_type = b[offset];
            let count = u32::from_be_bytes(b[offset+1..offset+5].try_into().ok()?) as usize;
            offset += 5;
            let mut items = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                if offset >= b.len() { break; }
                if let Some((v, next_off)) = parse_thrift_val(elem_type, b, offset) {
                    items.push(v);
                    offset = next_off;
                } else {
                    break;
                }
            }
            Some((ThriftVal::List(items), offset))
        }
        13 => { // MAP
            if offset + 6 > b.len() { return None; }
            let kt = b[offset];
            let vt = b[offset+1];
            let count = u32::from_be_bytes(b[offset+2..offset+6].try_into().ok()?) as usize;
            offset += 6;
            let mut pairs = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                if offset >= b.len() { break; }
                if let Some((k, off_k)) = parse_thrift_val(kt, b, offset) {
                    if let Some((v, off_v)) = parse_thrift_val(vt, b, off_k) {
                        pairs.push((k, v));
                        offset = off_v;
                    } else { break; }
                } else { break; }
            }
            Some((ThriftVal::Map(pairs), offset))
        }
        _ => None,
    }
}

pub fn parse_pokerstars_card(suit: i32, rank: i32) -> Option<Card> {
    let r = match rank {
        2 => Rank::Two,
        3 => Rank::Three,
        4 => Rank::Four,
        5 => Rank::Five,
        6 => Rank::Six,
        7 => Rank::Seven,
        8 => Rank::Eight,
        9 => Rank::Nine,
        10 => Rank::Ten,
        11 => Rank::Jack,
        12 => Rank::Queen,
        13 => Rank::King,
        14 => Rank::Ace,
        _ => return None,
    };
    let s = match suit {
        1 => Suit::Spades,
        2 => Suit::Hearts,
        3 => Suit::Diamonds,
        4 => Suit::Clubs,
        _ => return None,
    };
    Some(Card::new(r, s))
}

pub fn robust_parse_card(s: &str) -> Option<Card> {
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

pub fn parse_cards_array(val: &Value) -> Option<[Card; 2]> {
    let arr = val.as_array()?;
    let mut cards = Vec::new();
    for item in arr {
        if let Some(s) = item.as_str() { 
            if let Some(card) = robust_parse_card(s) { cards.push(card); } 
        } else if let Some(s) = item.get("value").and_then(|v| v.as_str()) { 
            if let Some(card) = robust_parse_card(s) { cards.push(card); } 
        }
    }
    if cards.len() == 2 { Some([cards[0], cards[1]]) } else { None }
}

// =========================================================================
// 3. WINDOW FOCUS & OS AUTOMATION (ENIGO)
// =========================================================================

static TARGET_GAME_ID: RwLock<String> = RwLock::new(String::new());

pub fn set_target_game(url_or_id: &str) {
    let clean_id = url_or_id.trim_matches('/').to_string();
    if let Ok(mut lock) = TARGET_GAME_ID.write() {
        *lock = clean_id;
    }
}

pub fn focus_pokernow_window() {
    #[cfg(target_os = "linux")]
    {
        let window_title = "pokerstars";
        let _ = std::process::Command::new("wmctrl")
            .args(["-a", window_title])
            .output();
    }
}

pub fn type_bet_amount(enigo: &mut Enigo, amount: u32) {
    let _ = enigo.key(Key::Unicode('r'), enigo::Direction::Click);
    thread::sleep(Duration::from_millis(100));
    for _ in 0..6 {
        let _ = enigo.key(Key::Backspace, enigo::Direction::Click);
        thread::sleep(Duration::from_millis(15));
    }
    for ch in amount.to_string().chars() {
        let _ = enigo.key(Key::Unicode(ch), enigo::Direction::Click);
        thread::sleep(Duration::from_millis(30));
    }
    thread::sleep(Duration::from_millis(50));
    let _ = enigo.key(Key::Return, enigo::Direction::Click);
}

pub fn execute_bot_action(action: Action, to_call: u32, target_total: u32, max_total_bet: u32, can_raise: bool) {
    focus_pokernow_window();
    thread::sleep(Duration::from_millis(150));

    let mut enigo = match Enigo::new(&Settings::default()) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("❌ [ENIGO] Không thể tạo Enigo keyboard driver: {:?}", err);
            return;
        }
    };

    if !can_raise && action != Action::Fold && action != Action::CheckCall {
        println!("⚠️ [BẢO VỆ] Bàn không thể Tố/All-In -> Tự động chuyển thành CHECK/CALL!");
        if to_call == 0 {
            let _ = enigo.key(Key::Unicode('k'), enigo::Direction::Click);
        } else {
            let _ = enigo.key(Key::Unicode('c'), enigo::Direction::Click);
        }
        return;
    }

    match action {
        Action::Fold => {
            println!("🤖 [POKERSTARS BOT] Thao tác: FOLD (Phím F)");
            let _ = enigo.key(Key::Unicode('f'), enigo::Direction::Click);
        }
        Action::CheckCall => {
            if to_call == 0 {
                println!("🤖 [POKERSTARS BOT] Thao tác: CHECK (Phím K)");
                let _ = enigo.key(Key::Unicode('k'), enigo::Direction::Click);
            } else {
                println!("🤖 [POKERSTARS BOT] Thao tác: CALL {} chip (Phím C)", to_call);
                let _ = enigo.key(Key::Unicode('c'), enigo::Direction::Click);
            }
        }
        Action::AllIn => {
            println!("🤖 [POKERSTARS BOT] Thao tác: ALL-IN {} chip", max_total_bet);
            type_bet_amount(&mut enigo, max_total_bet);
        }
        Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
            let final_bet = target_total.min(max_total_bet);
            if final_bet >= max_total_bet {
                println!("🤖 [POKERSTARS BOT] Đạt mức trần, đổi thành ALL-IN {} chip", max_total_bet);
                type_bet_amount(&mut enigo, max_total_bet);
            } else {
                println!("🤖 [POKERSTARS BOT] Thao tác: RAISE lên {} chip", final_bet);
                type_bet_amount(&mut enigo, final_bet);
            }
        }
    }
}

// =========================================================================
// 4. GTO & DEEP CFR DECISION ENGINE
// =========================================================================

pub fn run_gto_evaluation(
    state: &HandState, 
    hole: [Card; 2], 
    policy: &DeepCFRPolicy, 
    mode: DecisionMode, 
    _use_os_enigo: bool
) -> Option<InPageAction> {
    if state.bb_size == 0 {
        println!("⚠️ [CHỜ DỮ LIỆU SOCKET] Blinds = 0 -> Chưa nhận được cấu hình Blinds từ socket!");
        return None;
    }

    let mut sorted_players: Vec<&PlayerTracker> = state.players.values().collect();
    sorted_players.sort_by_key(|p| p.seat_num);

    let mut active_seats: Vec<String> = sorted_players
        .iter()
        .filter(|p| {
            let is_hero = (!state.hero_id.is_empty() && p.name.eq_ignore_ascii_case(&state.hero_id)) 
                || Some(p.seat_num) == state.hero_seat 
                || p.id == state.hero_id;
            
            if is_hero {
                return true;
            }

            // Loại bỏ người ngồi đợi blind hoặc tạm nghỉ mà chưa tham gia ván
            let is_waiting_or_sitting_out = (!p.is_in_hand || p.is_sitting_out) 
                && p.current_bet == 0 
                && p.total_contributed == 0 
                && !p.is_all_in;

            if is_waiting_or_sitting_out {
                return false;
            }

            // Người chơi hợp lệ của ván đấu: còn chip, có cược, đã góp chip, đã all-in, hoặc chưa fold
            p.stack > 0 || p.current_bet > 0 || p.total_contributed > 0 || p.is_all_in || !p.is_folded
        })
        .map(|p| p.id.clone())
        .collect();

    // 🛡️ CHỐT CHẶN BẢO VỆ 1: Nếu active_seats < 2, tự động tìm thêm đối thủ đang ngồi tại bàn
    if active_seats.len() < 2 {
        for p in &sorted_players {
            if !active_seats.contains(&p.id) {
                active_seats.push(p.id.clone());
                if active_seats.len() >= 2 {
                    break;
                }
            }
        }
    }

    if active_seats.is_empty() { return None; }

    let num_players = active_seats.len().clamp(2, 8);
    let hero_seat = active_seats.iter().position(|id| {
        if let Some(p) = state.players.get(id) {
            if Some(p.seat_num) == state.hero_seat {
                return true;
            }
            if !state.hero_id.is_empty() && p.name.eq_ignore_ascii_case(&state.hero_id) {
                return true;
            }
        }
        false
    }).unwrap_or(0);

    let mut actual_highest_bet = state.highest_bet;
    let mut sum_current_bets = 0;

    let mut initial_stacks = Vec::with_capacity(num_players);
    let mut env_players = Vec::with_capacity(num_players);

    for (seat_idx, pid) in active_seats.iter().enumerate() {
        let p_track = state.players.get(pid).cloned().unwrap_or_default();
        if p_track.current_bet > actual_highest_bet {
            actual_highest_bet = p_track.current_bet;
        }
        
        // Trong PokerStars socket (0x0076 & 0x006c), p_track.stack đã là stack còn lại (stack behind)
        let true_stack_behind = if p_track.is_all_in { 0 } else { p_track.stack };
        let starting_stack = true_stack_behind + p_track.current_bet;
        initial_stacks.push(starting_stack.max(state.bb_size));

        let mut ps = PlayerState::new(seat_idx, true_stack_behind, seat_idx != hero_seat);
        ps.current_bet = p_track.current_bet;
        ps.total_contributed = p_track.total_contributed.max(p_track.current_bet);
        ps.is_folded = if seat_idx == hero_seat { false } else { p_track.is_folded };
        ps.is_all_in = p_track.is_all_in;
        if seat_idx == hero_seat { ps.hole_cards = [Some(hole[0]), Some(hole[1])]; }
        
        sum_current_bets += p_track.current_bet;
        env_players.push(ps);
    }

    // 🛡️ CHỐT CHẶN BẢO VỆ 2: Đảm bảo 100% initial_stacks.len() == num_players (Triệt tiêu vĩnh viễn assertion failure!)
    while initial_stacks.len() < num_players {
        let opp_stack = (actual_highest_bet).max(state.bb_size * 50);
        initial_stacks.push(opp_stack);
        let dummy_seat = env_players.len();
        let mut ps = PlayerState::new(dummy_seat, opp_stack, true);
        ps.current_bet = actual_highest_bet;
        ps.total_contributed = actual_highest_bet;
        env_players.push(ps);
    }

    let in_hand_count = active_seats.iter().filter(|pid| {
        state.players.get(*pid).is_some_and(|pl| !pl.is_folded)
    }).count();

    let button_idx = if let Some(d_seat) = state.dealer_seat {
        active_seats.iter().position(|id| {
            state.players.get(id).map(|p| p.seat_num == d_seat).unwrap_or(false)
        }).unwrap_or(0)
    } else {
        0
    };
    let mut env = Environment::new(num_players, &initial_stacks, state.bb_size, state.sb_size, button_idx);

    let true_table_pot = state.total_pot.max(sum_current_bets).max(state.bb_size + state.sb_size);

    let mut current_bets: Vec<u32> = state.players.values()
        .map(|p| p.current_bet)
        .filter(|&b| b > 0)
        .collect();
    current_bets.sort_unstable();
    current_bets.dedup();

    let inferred_min_raise = if current_bets.is_empty() {
        state.bb_size
    } else if current_bets.len() == 1 {
        if state.street == Street::Preflop {
            state.bb_size
        } else {
            current_bets[0].max(state.bb_size)
        }
    } else {
        let highest = current_bets[current_bets.len() - 1];
        let second = current_bets[current_bets.len() - 2];
        (highest - second).max(state.bb_size)
    };

    let safe_min_raise = state.min_raise.max(inferred_min_raise).max(state.bb_size);
    
    env.state.pot = true_table_pot;
    env.state.current_highest_bet = actual_highest_bet;
    env.state.min_raise = safe_min_raise;
    env.state.community_cards = state.community_cards.clone();
    env.state.street = state.street;
    env.state.current_player_idx = hero_seat;
    env.state.button_idx = button_idx;
    env.state.players = env_players;
    env.state.is_cash_game = IS_CASH_GAME.load(Ordering::Relaxed);

    // ÁNH XẠ CHÍNH XÁC LỊCH SỬ CƯỢC VÀO ENGINE CHO MẠNG NƠ-RON HỌC VÀ TRÍCH XUẤT ĐẶC TRƯNG
    env.state.action_history = state.action_history.iter().filter_map(|rec| {
        let idx = active_seats.iter().position(|id| id == &rec.player_id)?;
        Some(ActionRecord {
            player_idx: idx,
            street: rec.street,
            action: rec.action,
            amount: rec.amount,
        })
    }).collect();

    let hero_player = &env.state.players[hero_seat];
    let to_call = actual_highest_bet.saturating_sub(hero_player.current_bet).min(hero_player.stack);

    let mut mask = env.get_action_mask();
    let features = env.state.encode_features(hero_seat);

    // 🛡️ CHỐT CHẶN ALL-IN AN TOÀN CASH GAME (TIER 1: EQUITY & SỨC MẠNH BÀI TẨY)
    // Trong PokerStars cash game (đặc biệt bàn 100 BB), All-in rủi ro toàn bộ tiền thật.
    // Chỉ cho phép nút All-In nếu bài có Equity mạnh hoặc Monster Value / Combo Draw thực thụ:
    let chen_strength = features[166];
    let eff_stack_bb = env.state.effective_stack_bb(hero_seat);
    let eff_chips = (eff_stack_bb * state.bb_size as f32).round() as u32;

    if mask.is_valid(Action::AllIn) {
        let is_all_in_allowed = if eff_stack_bb < 15.0 {
            // Stack ngắn (< 15 BB): cho phép All-in push/fold nếu bài tẩy đủ điểm Chen hoặc Equity khá
            if state.street == Street::Preflop {
                chen_strength >= 0.28
            } else {
                features[167] >= 0.40 || features[165] >= 0.22
            }
        } else {
            // Stack sâu (>= 15 BB, 50-100+ BB): Tiêu chuẩn All-in cực kỳ khắt khe!
            match state.street {
                Street::Preflop => {
                    // Preflop Deep-Stack:
                    // - Nếu stack >= 40 BB: Chỉ All-in bài siêu cấp (AA, KK, QQ, AKs - Chen >= 0.45)
                    // - Nếu stack 15-40 BB: Bài rất mạnh (TT+, AJs+, AKo - Chen >= 0.38)
                    if eff_stack_bb >= 40.0 {
                        chen_strength >= 0.45
                    } else {
                        chen_strength >= 0.38
                    }
                }
                Street::Flop | Street::Turn => {
                    // Postflop Deep-Stack:
                    // 1. Monster Made Hand: Two Pair trở lên (cat >= 0.30) hoặc Overpair Kicker khủng (cat >= 0.22 && str >= 0.75)
                    // 2. HOẶC Monster Combo Draw: Flush Draw + Straight Draw, hoặc raw equity >= 0.50
                    let hand_cat = features[165];
                    let hand_str = features[166];
                    let raw_eq = features[167];
                    let has_flush_draw = features[170] >= 0.5;
                    let has_straight_draw = features[171] >= 0.5;

                    let is_monster_made = hand_cat >= 0.30 || (hand_cat >= 0.22 && hand_str >= 0.75);
                    let is_monster_draw = (has_flush_draw && has_straight_draw) || raw_eq >= 0.50;

                    is_monster_made || is_monster_draw
                }
                Street::River => {
                    // River Deep-Stack: Không còn Draw. BẮT BUỘC Two Pair trở lên (cat >= 0.30)!
                    // Khóa cứng All-in nếu chỉ cầm One Pair hoặc High Card ở River!
                    features[166] >= 0.30
                }
                Street::Showdown => false,
            }
        };

        if !is_all_in_allowed && (mask.is_valid(Action::Fold) || mask.is_valid(Action::CheckCall) || mask.is_valid(Action::BetPot33) || mask.is_valid(Action::BetPot75) || mask.is_valid(Action::BetSmall)) {
            println!(
                "🛡️  [CASH ALL-IN GUARD] Bài chưa đủ điều kiện Equity mạnh ({:.1} BB, {:?}, Cat: {:.2}, Chen: {:.2}) -> Khóa All-In!",
                eff_stack_bb, state.street, features[166], chen_strength
            );
            mask.0 &= !(1 << Action::AllIn.to_index());
        }
    }

    // Chốt chặn phụ chống bluff bài rác postflop (Air)
    if state.street != Street::Preflop {
        let hand_cat = features[166];
        let has_flush_draw = features[171] >= 0.5;
        let has_straight_draw = features[172] >= 0.5;
        let is_pure_air = hand_cat < 0.15 && !has_flush_draw && !has_straight_draw;

        if to_call > 0 && is_pure_air {
            if mask.is_valid(Action::Fold) || mask.is_valid(Action::CheckCall) {
                println!(
                    "🛡️  [POSTFLOP ANTI-SUICIDE] Bài tẩy {:?} không có đôi / không có draw (Air) khi đối thủ bet -> Khóa toàn bộ Tố / All-In!",
                    hole
                );
                mask.0 &= !((1 << Action::BetSmall.to_index())
                    | (1 << Action::BetPot33.to_index())
                    | (1 << Action::BetPot75.to_index())
                    | (1 << Action::AllIn.to_index()));
            }
        }

        if in_hand_count >= 3 && is_pure_air {
            println!(
                "👥 [MULTIWAY AIR GUARD] Pot đông người ({} người) với bài rác -> Khóa BetPot75 & All-In!",
                in_hand_count
            );
            mask.0 &= !((1 << Action::BetPot75.to_index()) | (1 << Action::AllIn.to_index()));
        }
    }

    let mut clean_probs = policy.get_action_probs(&features, &mask);
    let raw_adv = policy.advantage_net.read().unwrap().forward(&features);

    let num_acts = raw_adv.len().min(clean_probs.len());
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
                clean_probs[a_idx] = e; sum_exp += e;
            } else { clean_probs[a_idx] = 0.0; }
        }
        if sum_exp > 0.0 { for p in &mut clean_probs { *p /= sum_exp; } }
    }

    let (chosen_idx, decision_note) = match mode {
        DecisionMode::PureMaxEv => {
            // 1. Tìm action có tần suất GTO cao nhất (max_gto_p)
            let mut max_gto_p = 0.0f32;
            let mut best_gto_idx = 0;
            for (i, &p) in clean_probs.iter().enumerate().take(NUM_ACTIONS) {
                if mask.is_index_valid(i) && p > max_gto_p {
                    max_gto_p = p;
                    best_gto_idx = i;
                }
            }

            // 2. Lọc các action có tần suất GTO đáng tin cậy:
            // - Với Action thông thường: p >= 8% và p >= max_gto_p * 0.30
            // - 🛡️ ĐẶC BIỆT VỚI ALL-IN (TIER 2: STRATEGY % & ADVANTAGE EV CONFIRMATION):
            //   Chỉ chấp nhận All-In nếu:
            //   a) Mạng Strategy cho % cao: clean_probs[AllIn] >= 25% VÀ >= max_gto_p * 0.60
            //   b) Mạng Advantage cho EV cao tuyệt đối: raw_adv[AllIn] > 0.10 (tránh EV dương ảo sát 0)
            let min_gto_threshold = 0.08f32;
            let mut candidates = Vec::new();
            for (i, &ev) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
                if !mask.is_index_valid(i) { continue; }
                let p = clean_probs[i];
                let act = Action::from_index(i).unwrap();

                if act == Action::AllIn {
                    let has_high_gto = p >= 0.25 && p >= max_gto_p * 0.60;
                    let has_strong_ev = ev > 0.10;
                    if has_high_gto && has_strong_ev {
                        candidates.push((i, ev, p));
                    } else {
                        println!(
                            "🛡️  [TIER 2 SHIELD] Loại All-In khỏi ứng viên do chưa đạt GTO/EV chuẩn (GTO: {:.1}%, EV: {:>+6.2})",
                            p * 100.0, ev
                        );
                    }
                } else if p >= min_gto_threshold && p >= max_gto_p * 0.30 {
                    candidates.push((i, ev, p));
                }
            }

            if candidates.is_empty() {
                let ev = raw_adv.get(best_gto_idx).copied().unwrap_or(0.0);
                candidates.push((best_gto_idx, ev, max_gto_p));
            }

            // Chọn ứng viên có EV cao nhất trong danh sách GTO hợp lệ
            let mut best_i = candidates[0].0;
            let mut best_ev = candidates[0].1;
            let mut best_p = candidates[0].2;
            for &(idx, ev, p) in &candidates[1..] {
                if ev > best_ev {
                    best_i = idx;
                    best_ev = ev;
                    best_p = p;
                }
            }

            let best_act = Action::from_index(best_i).unwrap();
            let note = if best_i == best_gto_idx {
                format!(
                    "⚡ [GTO + EV MAX] Tuyến cược GTO chủ đạo: {:?} (Tần suất GTO: {:.1}%, EV: {:>+6.2})",
                    best_act, best_p * 100.0, best_ev
                )
            } else {
                format!(
                    "⚡ [GTO + EV MAX] Chọn nhánh EV cao hơn trong các tuyến GTO khả thi: {:?} (EV: {:>+6.2}, GTO: {:.1}%)",
                    best_act, best_ev, best_p * 100.0
                )
            };
            (best_i, note)
        }
        _ => (Action::CheckCall.to_index(), "🎲 [GTO DEFAULT]".to_string()),
    };

    let max_total_bet = hero_player.current_bet + hero_player.stack;

    println!("\n============================================================");
    println!("🎯 [ĐẾN LƯỢT BẠN] - VÒNG: {:?}", state.street);
    if in_hand_count <= 2 {
        println!("⚔️  [ĐỐI ĐẦU 1-VS-1 HEADS-UP] Chỉ còn 2 người trong ván bài!");
    } else {
        println!("👥 Số người còn trong ván: {} / Tổng số ghế: {}", in_hand_count, num_players);
    }
    println!("🃏 Bài tẩy : [ {} {} ]", hole[0], hole[1]);
    if !state.community_cards.is_empty() {
        let board_str: Vec<String> = state.community_cards.iter().map(|c| format!("{}", c)).collect();
        println!("🎴 Bài chung: [ {} ]", board_str.join(" "));
    }
    
    let call_status = if to_call == 0 { "(Đang check)" } else { "" };
    if state.rake > 0 {
        println!("💰 Tổng Pot thực tế: {} chip (Rake: {} chip | Tổng góp: {} chip) | Cần Call: {} chip {} | Blind: {}/{} chip", 
            true_table_pot, state.rake, true_table_pot + state.rake, to_call, call_status, state.sb_size, state.bb_size);
    } else {
        println!("💰 Tổng Pot thực tế: {} chip | Cần Call: {} chip {} | Blind: {}/{} chip", true_table_pot, to_call, call_status, state.sb_size, state.bb_size);
    }
    println!("⚡ Effective Stack : {} chip ({:.1} BB)", eff_chips, eff_stack_bb);

    println!("🏦 [STACK CHIP THỰC TẾ]:");
    for pid in &active_seats {
        if let Some(p) = state.players.get(pid) {
            let is_hero = pid == &state.hero_id || Some(p.seat_num) == state.hero_seat;
            let display_name = if !p.name.is_empty() && p.name != "Player" {
                p.name.clone()
            } else {
                format!("Ghế #{}", p.seat_num)
            };
            let who = if is_hero { format!("{} (Hero)", display_name) } else { display_name };
            let true_stack = if p.is_all_in {
                0
            } else {
                p.stack.saturating_sub(p.current_bet)
            };
            let stack_bb = if state.bb_size > 0 { true_stack as f32 / state.bb_size as f32 } else { 0.0 };
            let is_folded = if is_hero { false } else { p.is_folded };
            let status_str = if p.is_all_in {
                if p.current_bet > 0 {
                    format!(" [ALL-IN {} chip]", p.current_bet)
                } else {
                    " [ALL-IN]".to_string()
                }
            } else if is_folded {
                " [FOLD]".to_string()
            } else if p.current_bet > 0 {
                format!(" (Đang cược {} chip)", p.current_bet)
            } else {
                String::new()
            };
            println!("   • Ghế #{:<2} {:<16} : {:>6} chip ({:>5.1} BB){}", 
                p.seat_num, who, true_stack, stack_bb, status_str);
        }
    }
    
    if !state.action_history.is_empty() {
        println!("📜 [LỊCH SỬ CƯỢC - ACTION HISTORY] ({} hành động):", state.action_history.len());
        for (idx, rec) in state.action_history.iter().enumerate() {
            let p_name = state.players.get(&rec.player_id)
                .map(|p| p.name.as_str())
                .unwrap_or("Unknown");
            let is_hero = rec.player_id == state.hero_id || Some(rec.seat_num) == state.hero_seat;
            let who = if is_hero { format!("{} (Hero)", p_name) } else { p_name.to_string() };
            let act_desc = match rec.action {
                Action::AllIn => format!("All-In (+{} chip)", rec.amount),
                Action::Fold => "Fold".to_string(),
                Action::CheckCall => {
                    if rec.amount == 0 { "Check".to_string() } else { format!("Call (+{} chip)", rec.amount) }
                }
                _ => format!("Tố (+{} chip)", rec.amount),
            };
            println!("   {}. [{:?}] {:<12} (Ghế #{:<2}): {}", 
                idx + 1, rec.street, who, rec.seat_num, act_desc);
        }
    }

    let mut active_p_list: Vec<&PlayerTracker> = state.players.values().collect();
    active_p_list.sort_by_key(|p| p.seat_num);
    let mut contrib_str = Vec::new();
    for p in active_p_list {
        let total = p.total_contributed.max(p.current_bet);
        if total > 0 {
            let display_name = if !p.name.is_empty() && p.name != "Player" {
                p.name.clone()
            } else {
                format!("Ghế #{}", p.seat_num)
            };
            let label = if p.id == state.hero_id || Some(p.seat_num) == state.hero_seat {
                format!("{} (Hero)", display_name)
            } else {
                display_name
            };
            contrib_str.push(format!("{}: {} chip", label, total));
        }
    }
    if !contrib_str.is_empty() {
        println!("💵 [TỔNG CHIP ĐÓNG GÓP]: {}", contrib_str.join(" | "));
    }
    println!("------------------------------------------------------------");
    println!("📊 BẢNG ĐÁNH GIÁ NƯỚC ĐI [THUẦN EV MAX]:");

    let pot_after_call = true_table_pot + to_call;
    let bet_small_target = (actual_highest_bet + (((pot_after_call as f32) * 0.15).round() as u32).max(safe_min_raise)).min(max_total_bet);
    let bet_33_target = (actual_highest_bet + (((pot_after_call as f32) * 0.33).round() as u32).max(safe_min_raise)).min(max_total_bet);
    let is_small_same_as_33 = bet_small_target == bet_33_target;

    for (i, &p) in clean_probs.iter().enumerate().take(NUM_ACTIONS) {
        let act = Action::from_index(i).unwrap();
        let is_valid = mask.is_index_valid(i);
        let raw_val = raw_adv.get(i).copied().unwrap_or(0.0);

        let chip_desc = match act {
            Action::Fold => "0".to_string(),
            Action::CheckCall => {
                if to_call == 0 { "Check".to_string() } else { format!("Call {}", to_call) }
            }
            Action::AllIn => format!("All-In ({})", max_total_bet),
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                let frac = match act {
                    Action::BetSmall => 0.15,
                    Action::BetPot33 => 0.33,
                    Action::BetPot75 => 0.75,
                    _ => unreachable!(),
                };
                let raise_by = (((pot_after_call as f32) * frac).round() as u32).max(safe_min_raise);
                let target = (actual_highest_bet + raise_by).min(max_total_bet);
                if actual_highest_bet == 0 {
                    if raise_by == safe_min_raise && frac == 0.15 {
                        format!("Cược {} (Min Bet)", target)
                    } else {
                        format!("Cược {}", target)
                    }
                } else {
                    if raise_by == safe_min_raise && frac == 0.15 {
                        format!("Tố lên {} (Min Raise)", target)
                    } else {
                        format!("Tố lên {}", target)
                    }
                }
            }
        };

        let rec = if is_valid && i == chosen_idx { " ⭐ TỐI ĐA EV (ƯU TIÊN GTO)" } else { "" };
        let status = if is_valid {
            ""
        } else {
            if act == Action::Fold && to_call == 0 {
                " (Khóa: Check miễn phí)"
            } else if act == Action::BetSmall && is_small_same_as_33 {
                " (Khóa: Trùng size với Bet 33%)"
            } else {
                " (Bị khóa do luật cược / không đủ chip)"
            }
        };

        println!("    {:<14}: EV: {:>+6.2} (Tần suất GTO: {:>5.1}%) | [{:<15}]{}{}",
            act.name(), raw_val, p * 100.0, chip_desc, rec, status);
    }

    if let Some(best_action) = Action::from_index(chosen_idx) {
        println!("{}", decision_note);
        println!("👉 Quyết định cuối cùng: {:?} (EV: {:>+6.2})", best_action, raw_adv.get(chosen_idx).copied().unwrap_or(0.0));

        let mut target_total = actual_highest_bet;
        match best_action {
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                let frac = match best_action {
                    Action::BetSmall => 0.15,
                    Action::BetPot33 => 0.33,
                    Action::BetPot75 => 0.75,
                    _ => unreachable!(),
                };
                let pot_after_call = true_table_pot + to_call;
                let raise_by = (((pot_after_call as f32) * frac).round() as u32).max(safe_min_raise);
                target_total = (actual_highest_bet + raise_by).min(max_total_bet);
            }
            Action::AllIn => {
                target_total = max_total_bet;
            }
            _ => {}
        }

        let can_raise = hero_player.stack > to_call && in_hand_count >= 2;
        let safe_action = if !can_raise && best_action != Action::Fold && best_action != Action::CheckCall {
            Action::CheckCall
        } else {
            best_action
        };

        // Độ trễ cược theo yêu cầu:
        // - Check/Call/Fold: Không trễ (0 ms) -> Bấm ngay lập tức (snap check/call/fold).
        // - Raise / Bet / All-in: Giảm bớt 1.5s so với mức cũ (1500..=3000 ms) -> Mức mới: 200..=1500 ms (0.2s - 1.5s)
        let delay_ms: u64 = match safe_action {
            Action::Fold | Action::CheckCall => 0,
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 | Action::AllIn => {
                rand::thread_rng().gen_range(200..=1500)
            }
        };

        let in_page_cmd = match safe_action {
            Action::Fold => InPageAction {
                action: "fold".to_string(),
                amount: 0,
                to_call: 0,
                delay_ms,
            },
            Action::CheckCall => {
                if to_call == 0 {
                    InPageAction {
                        action: "check".to_string(),
                        amount: 0,
                        to_call: 0,
                        delay_ms,
                    }
                } else {
                    InPageAction {
                        action: "call".to_string(),
                        amount: to_call,
                        to_call,
                        delay_ms,
                    }
                }
            }
            Action::AllIn => InPageAction {
                action: "all_in".to_string(),
                amount: max_total_bet,
                to_call,
                delay_ms,
            },
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                let final_bet = target_total.min(max_total_bet);
                if final_bet >= max_total_bet {
                    InPageAction {
                        action: "all_in".to_string(),
                        amount: max_total_bet,
                        to_call,
                        delay_ms,
                    }
                } else {
                    InPageAction {
                        action: "raise".to_string(),
                        amount: final_bet,
                        to_call,
                        delay_ms,
                    }
                }
            }
        };

        let delay_str = if delay_ms == 0 {
            "Tức thì (0s)".to_string()
        } else {
            format!("{:.2}s", delay_ms as f64 / 1000.0)
        };
        println!(
            "🥷 [AUTO-BOT] Đã gửi lệnh tự bấm sang PokerStars Web: {:?} (Số chip: {}, Trễ: {})",
            in_page_cmd.action.to_uppercase(),
            in_page_cmd.amount,
            delay_str
        );
        println!("============================================================\n");
        Some(in_page_cmd)
    } else {
        println!("============================================================\n");
        None
    }
}

// =========================================================================
// 5. POKERSTARS THRIFT PROTOCOL HANDLER
// =========================================================================

pub fn extract_cards_from_thrift(val: &ThriftVal, out: &mut Vec<Card>) {
    match val {
        ThriftVal::Struct(fields) => {
            // Cấu trúc lá bài PokerStars: { 1: id/0/255, 2: suit (1..=4), 3: rank (2..=14) } và KHÔNG có field 4 (tránh nhầm với gói action 0x0072)
            if (fields.contains_key(&1) || fields.len() <= 3) && !fields.contains_key(&4) {
                if let (Some(s_val), Some(r_val)) = (fields.get(&2), fields.get(&3)) {
                    if let (Some(s), Some(r)) = (s_val.get_i32(), r_val.get_i32()) {
                        if (0..=4).contains(&s) && (0..=14).contains(&r) {
                            if let Some(c) = parse_pokerstars_card(s, r) {
                                out.push(c);
                            }
                        }
                    }
                }
            }
            for v in fields.values() {
                extract_cards_from_thrift(v, out);
            }
        }
        ThriftVal::List(list) => {
            for v in list {
                extract_cards_from_thrift(v, out);
            }
        }
        ThriftVal::Map(map) => {
            for (k, v) in map {
                extract_cards_from_thrift(k, out);
                extract_cards_from_thrift(v, out);
            }
        }
        _ => {}
    }
}

pub fn process_pokerstars_thrift_packet(
    state: &mut HandState,
    opcode: u16,
    thrift_data: &ThriftVal,
    policy: &DeepCFRPolicy,
    decision_mode: DecisionMode,
    use_os_enigo: bool,
) -> Option<InPageAction> {
    // Tự động nhận diện ván mới khi Hand ID thay đổi từ socket (hỗ trợ cả Cash 6R, Play 6P và Zoom xoay vòng 6Z)
    if let Some(h_id) = thrift_data.get_field(1).and_then(|v| v.get_str()) {
        let is_valid_hand_topic = (h_id.starts_with("6R.") || h_id.starts_with("6P.") || h_id.starts_with("6Z.")) 
            && h_id.matches('.').count() >= 2;
        if !h_id.is_empty() && is_valid_hand_topic && h_id != state.hand_id {
            state.start_new_hand(h_id);
        }
    }

    // Quét tìm 2 lá bài tẩy trong bất kỳ gói tin nào gửi riêng cho Hero (khác opcode 0x0071 bài chung)
    if opcode != 0x0071 {
        let mut found_cards = Vec::new();
        extract_cards_from_thrift(thrift_data, &mut found_cards);
        if found_cards.len() == 2 && (opcode == 0x006c || opcode == 0x008b || opcode == 0x005a) {
            let is_hero_pkt = if opcode == 0x006c {
                if let Some(p_data) = thrift_data.get_field(2) {
                    let seat = p_data.get_field(1).and_then(|v| v.get_i32()).unwrap_or(-1) as usize;
                    Some(seat) == state.hero_seat
                } else {
                    false
                }
            } else {
                true
            };

            if is_hero_pkt {
                state.hero_cards = Some([found_cards[0], found_cards[1]]);
                println!("🃏 [NHẬN DIỆN BÀI TẨY HERO TỪ SOCKET (0x{:04x})]: [ {}  {} ]", opcode, found_cards[0], found_cards[1]);
            }
        }
    }

    match opcode {
        0x006d => { // KHỞI TẠO VÁN BÀI MỚI (NEW HAND INITIALIZATION)
            let hand_id = thrift_data.get_field(2).and_then(|v| v.get_str()).unwrap_or("");
            let dealer = thrift_data.get_field(3).and_then(|v| v.get_i32());
            if !hand_id.is_empty() && hand_id != state.hand_id {
                state.start_new_hand(hand_id);
                if let Some(d) = dealer {
                    state.dealer_seat = Some(d as usize);
                    println!("🔘 [DEALER BUTTON]: Ghế #{}", d);
                }
            }
        }
        0x006a => { // THÔNG TIN BÀN CHƠI (TABLE INFO)
            if let Some(t_info) = thrift_data.get_field(2) {
                let name = t_info.get_field(1).and_then(|v| v.get_str()).unwrap_or("?");
                let seats = t_info.get_field(4).and_then(|v| v.get_i32()).unwrap_or(6);

                // Đọc Blinds hoàn toàn từ Socket (Field 7 hoặc Field 15) - KHÔNG HARDCODE
                let mut sb = 0u32;
                let mut bb = 0u32;

                if let Some(b_struct) = t_info.get_field(7) {
                    let raw_sb = b_struct.get_field(1).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
                    let raw_bb = b_struct.get_field(2).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
                    sb = raw_sb / 100;
                    bb = raw_bb / 100;
                }
                if bb == 0 {
                    if let Some(b_list) = t_info.get_field(15).and_then(|v| v.get_field(2)).and_then(|v| v.get_list()) {
                        if b_list.len() >= 2 {
                            let raw_sb = b_list[0].get_i32().unwrap_or(0).max(0) as u32;
                            let raw_bb = b_list[1].get_i32().unwrap_or(0).max(0) as u32;
                            sb = raw_sb / 100;
                            bb = raw_bb / 100;
                        }
                    }
                }

                state.table_name = name.to_string();
                if bb > 0 {
                    state.sb_size = sb;
                    state.bb_size = bb;
                    state.min_raise = bb;
                    println!("🏛️ [SOCKET 0x006a]: Bàn \"{}\" ({}-Max) | Blinds từ socket: {} / {} chip (Min Raise = {})",
                        name, seats, sb, bb, bb);
                } else {
                    println!("🏛️ [SOCKET 0x006a]: Bàn \"{}\" ({}-Max) | Đang chờ socket cập nhật Blinds...", name, seats);
                }
            }

            // Dealer button seat
            if let Some(dealer) = thrift_data.get_field(7).and_then(|v| v.get_i32()) {
                state.dealer_seat = Some(dealer as usize);
                println!("🔘 [DEALER BUTTON]: Ghế #{}", dealer);
            }

            // Reset trạng thái bàn chơi cho bàn mới (đặc biệt trong chế độ xoay vòng Zoom)
            state.community_cards.clear();
            state.action_history.clear();
            state.street = Street::Preflop;
            state.highest_bet = 0;
            state.total_pot = 0;
            state.has_evaluated_this_decision = false;
            state.current_turn_seat = None;
            state.players.clear();

            // Danh sách toàn bộ người chơi tại bàn khi mở bàn (Field 4)
            if let Some(player_list) = thrift_data.get_field(4).and_then(|v| v.get_list()) {
                for p_data in player_list {
                    let seat = p_data.get_field(1).and_then(|v| v.get_i32()).unwrap_or(-1) as usize;
                    let seat_status = p_data.get_field(2).and_then(|v| v.get_i32()).unwrap_or(0);
                    let pid = format!("seat_{}", seat);

                    if seat_status == 0 {
                        state.players.remove(&pid);
                    } else if let Some(info) = p_data.get_field(3) {
                        let pname = info.get_field(1).and_then(|v| v.get_str()).unwrap_or("?").to_string();
                        let country = info.get_field(2).and_then(|v| v.get_str()).unwrap_or("??").to_string();
                        let is_sitting_in = match info.get_field(4) {
                            Some(ThriftVal::Bool(b)) => *b,
                            _ => true,
                        };
                        let is_in_hand = match info.get_field(5) {
                            Some(ThriftVal::Bool(b)) => *b,
                            _ => true,
                        };
                        let raw_stack = info.get_field(7).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
                        let stack = raw_stack / 100;
                        let raw_cur_bet = p_data.get_field(4).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
                        let cur_bet = raw_cur_bet / 100;

                        if !state.hero_id.is_empty() && pname.eq_ignore_ascii_case(&state.hero_id) {
                            state.hero_seat = Some(seat);
                            println!("🎯 [HERO NHẬN DIỆN]: Ghế #{} là Hero [{}]!", seat, pname);
                        } else if state.hero_id.is_empty() && state.hero_seat == Some(seat) && !pname.is_empty() && pname != "Player" {
                            state.hero_id = pname.clone();
                            println!("🎯 [HERO NHẬN DIỆN]: Ghế #{} là Hero [{}]!", seat, pname);
                        }

                        let tracker = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                            id: pid.clone(),
                            name: pname.clone(),
                            seat_num: seat,
                            stack,
                            current_bet: cur_bet,
                            total_contributed: cur_bet,
                            is_folded: false,
                            is_all_in: false,
                            is_sitting_out: !is_sitting_in,
                            is_in_hand,
                        });
                        tracker.name = pname.clone();
                        tracker.seat_num = seat;
                        tracker.stack = stack;
                        tracker.current_bet = cur_bet;
                        tracker.is_sitting_out = !is_sitting_in;
                        tracker.is_in_hand = is_in_hand;

                        let status_note = if !is_sitting_in {
                            " | ⏸️ [TẠM NGHỈ / SITTING OUT]"
                        } else if !is_in_hand {
                            " | ⏳ [ĐỢI BLIND / CHƯA CHƠI]"
                        } else {
                            ""
                        };
                        println!("👤 [BÀN CHƠI] Ghế #{}: {:<14} ({}) | Stack: {:>8} chip | Cược: {:>6} chip{}", seat, pname, country, stack, cur_bet, status_note);
                    }
                }
            }
        }
        0x006c => { // CẬP NHẬT GHẾ & NGƯỜI CHƠI (PLAYER SEAT)
            if let Some(p_data) = thrift_data.get_field(2) {
                let seat = p_data.get_field(1).and_then(|v| v.get_i32()).unwrap_or(-1) as usize;
                let seat_status = p_data.get_field(2).and_then(|v| v.get_i32()).unwrap_or(0);
                let pid = format!("seat_{}", seat);

                if seat_status == 0 {
                    state.players.remove(&pid);
                    println!("🚪 [RỜI BÀN] Ghế #{}: Đã rời bàn / ghế trống", seat);
                } else if let Some(info) = p_data.get_field(3) {
                    let pname = info.get_field(1).and_then(|v| v.get_str()).unwrap_or("?").to_string();
                    let country = info.get_field(2).and_then(|v| v.get_str()).unwrap_or("??").to_string();
                    let is_sitting_in = match info.get_field(4) {
                        Some(ThriftVal::Bool(b)) => *b,
                        _ => true,
                    };
                    let is_in_hand = match info.get_field(5) {
                        Some(ThriftVal::Bool(b)) => *b,
                        _ => true,
                    };
                    let raw_stack = info.get_field(7).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
                    let stack = raw_stack / 100;
                    let raw_cur_bet = p_data.get_field(4).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
                    let cur_bet = raw_cur_bet / 100;

                    if !state.hero_id.is_empty() && pname.eq_ignore_ascii_case(&state.hero_id) {
                        state.hero_seat = Some(seat);
                        println!("🎯 [HERO NHẬN DIỆN]: Ghế #{} là Hero [{}]!", seat, pname);
                    } else if state.hero_id.is_empty() && state.hero_seat == Some(seat) && !pname.is_empty() && pname != "Player" {
                        state.hero_id = pname.clone();
                        println!("🎯 [HERO NHẬN DIỆN]: Ghế #{} là Hero [{}]!", seat, pname);
                    }

                    let tracker = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                        id: pid.clone(),
                        name: pname.clone(),
                        seat_num: seat,
                        stack,
                        current_bet: cur_bet,
                        total_contributed: cur_bet,
                        is_folded: false,
                        is_all_in: stack == 0 && cur_bet > 0,
                        is_sitting_out: !is_sitting_in,
                        is_in_hand,
                    });
                    tracker.name = pname.clone();
                    tracker.seat_num = seat;
                    tracker.stack = stack;
                    if cur_bet > tracker.current_bet {
                        tracker.current_bet = cur_bet;
                    }
                    tracker.is_sitting_out = !is_sitting_in;
                    if !tracker.is_folded && (tracker.current_bet > 0 || tracker.total_contributed > 0 || tracker.is_all_in) {
                        tracker.is_in_hand = true;
                    } else {
                        tracker.is_in_hand = is_in_hand;
                    }
                    if stack == 0 && (tracker.current_bet > 0 || tracker.total_contributed > 0) {
                        tracker.is_all_in = true;
                    }

                    let status_note = if !is_sitting_in {
                        " | ⏸️ [TẠM NGHỈ / SITTING OUT]"
                    } else if !is_in_hand {
                        " | ⏳ [ĐỢI BLIND / CHƯA CHƠI]"
                    } else {
                        ""
                    };
                    println!("👤 [NGƯỜI CHƠI] Ghế #{}: {:<14} ({}) | Stack: {:>8} chip | Cược: {:>6} chip{}", seat, pname, country, stack, cur_bet, status_note);
                }
            }
        }
        0x0072 => { // HÀNH ĐỘNG CỦA NGƯỜI CHƠI (PLAYER ACTION)
            let seat = thrift_data.get_field(2).and_then(|v| v.get_i32()).unwrap_or(-1) as usize;
            let action_type = thrift_data.get_field(3).and_then(|v| v.get_i32()).unwrap_or(0);
            let raw_amount = thrift_data.get_field(4).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
            let amount = raw_amount / 100;
            let pid = format!("seat_{}", seat);

            let act_desc = match action_type {
                3 => "POST SB",
                4 => "POST BB",
                5 => "POST ANTE",
                7 | 10 => "FOLD",
                11 => "CHECK",
                12 => "CALL",
                13 => "BET",
                14 => "ALL-IN",
                15 => "RAISE",
                21 => "MUCK/SHOW",
                22 => "WIN/MUCK",
                _ => "ACTION",
            };

            if let Some(p) = state.players.get_mut(&pid) {
                let prev_bet = p.current_bet;
                let delta = amount.saturating_sub(prev_bet);

                if action_type == 10 || action_type == 7 {
                    p.is_folded = true;
                    if Some(seat) == state.hero_seat {
                        state.hero_cards = None;
                        state.has_evaluated_this_decision = false;
                        state.action_history.clear();
                        state.community_cards.clear();
                        state.street = Street::Preflop;
                        state.total_pot = 0;
                        state.highest_bet = 0;
                        state.current_turn_seat = None;
                        println!("🛡️ [HERO FOLD / ZOOM RESET] Hero đã Fold -> Đã reset trạng thái ván cũ!");
                    } else {
                        state.has_evaluated_this_decision = false;
                    }
                    state.action_history.push(HandActionRecord {
                        player_id: pid.clone(),
                        seat_num: seat,
                        street: state.street,
                        action: Action::Fold,
                        amount: 0,
                    });
                } else if action_type == 11 {
                    p.is_folded = false;
                    if Some(seat) != state.hero_seat {
                        state.has_evaluated_this_decision = false;
                    }
                    state.action_history.push(HandActionRecord {
                        player_id: pid.clone(),
                        seat_num: seat,
                        street: state.street,
                        action: Action::CheckCall,
                        amount: 0,
                    });
                } else if action_type == 12 { // CALL
                    p.is_folded = false;
                    p.current_bet = amount;
                    p.total_contributed += delta;
                    if Some(seat) != state.hero_seat {
                        state.has_evaluated_this_decision = false;
                    }
                    if p.stack == 0 || (p.stack > 0 && delta >= p.stack) {
                        p.is_all_in = true;
                        println!("🚨 [CALL ALL-IN DETECTED] Ghế #{} [{}] Call hết chip -> Đánh dấu ALL-IN!", seat, p.name);
                    }
                    state.action_history.push(HandActionRecord {
                        player_id: pid.clone(),
                        seat_num: seat,
                        street: state.street,
                        action: if p.is_all_in { Action::AllIn } else { Action::CheckCall },
                        amount: delta,
                    });
                } else if action_type == 13 || action_type == 14 || action_type == 15 {
                    p.is_folded = false;
                    p.current_bet = amount;
                    p.total_contributed += delta;
                    if Some(seat) != state.hero_seat {
                        state.has_evaluated_this_decision = false;
                    }
                    if action_type == 14 || p.stack == 0 || (p.stack > 0 && delta >= p.stack) {
                        p.is_all_in = true;
                        println!("🚨 [ALL-IN SHOVE DETECTED] Ghế #{} [{}] Tố hết chip -> Đánh dấu ALL-IN!", seat, p.name);
                    }

                    if amount > state.highest_bet {
                        let raise_diff = amount - state.highest_bet;
                        if raise_diff >= state.min_raise {
                            state.min_raise = raise_diff;
                        }
                        state.highest_bet = amount;
                    }

                    let act_enum = if action_type == 14 || p.is_all_in {
                        Action::AllIn
                    } else if delta <= state.bb_size * 2 && state.street != Street::Preflop {
                        Action::BetSmall
                    } else if (delta as f32) < (state.total_pot.max(state.bb_size * 2) as f32 * 0.50) {
                        Action::BetPot33
                    } else {
                        Action::BetPot75
                    };

                    state.action_history.push(HandActionRecord {
                        player_id: pid.clone(),
                        seat_num: seat,
                        street: state.street,
                        action: act_enum,
                        amount: delta,
                    });
                } else if action_type == 3 || action_type == 4 || action_type == 5 { // BLINDS / ANTE
                    p.is_folded = false;
                    p.current_bet = amount;
                    p.total_contributed += delta;
                    if Some(seat) != state.hero_seat {
                        state.has_evaluated_this_decision = false;
                    }
                    if amount > state.highest_bet {
                        state.highest_bet = amount;
                    }
                }
                println!("🎬 [HÀNH ĐỘNG BÀN] Ghế #{}: {} (Mức cược: {} chip)", seat, act_desc, amount);
            }
        }
        0x0078 => { // TỔNG TIỀN GÀ (POT & RAKE)
            if let Some(pots) = thrift_data.get_field(2).and_then(|v| v.get_list()) {
                // ID 10 là Tổng Net Pot, ID 0 là Main Pot, ID 11 là Tiền Rake PokerStars thu
                let mut net_pot = 0u32;
                let mut rake_val = 0u32;
                for p in pots {
                    let pot_id = p.get_field(1).and_then(|v| v.get_i32()).unwrap_or(-1);
                    let amt = (p.get_field(2).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32) / 100;
                    if pot_id == 11 {
                        rake_val = amt;
                    } else if pot_id == 10 {
                        net_pot = amt;
                    } else if pot_id == 0 && net_pot == 0 {
                        net_pot = amt;
                    }
                }
                if net_pot > 0 {
                    state.total_pot = net_pot;
                }
                state.rake = rake_val;
                if rake_val > 0 {
                    println!("🏆 [TỔNG POT]: {} chip (Rake bàn: {} chip | Tổng tiền góp: {} chip)",
                        state.total_pot, rake_val, state.total_pot + rake_val);
                } else {
                    println!("🏆 [TỔNG POT]: {} chip", state.total_pot);
                }
            }
        }
        0x0079 => { // MẢNG CƯỢC TRÊN BÀN (TABLE BETS)
            if let Some(bets) = thrift_data.get_field(3).and_then(|v| v.get_list()) {
                let mut max_b = 0;
                let mut second_b = 0;
                for (seat_idx, bet_val) in bets.iter().enumerate() {
                    let b = (bet_val.get_i32().unwrap_or(0).max(0) as u32) / 100;
                    if b > max_b {
                        second_b = max_b;
                        max_b = b;
                    } else if b > second_b {
                        second_b = b;
                    }
                    let pid = format!("seat_{}", seat_idx);
                    if let Some(p) = state.players.get_mut(&pid) {
                        p.current_bet = b;
                    }
                }
                state.highest_bet = max_b;

                // Tự động suy luận Blinds từ cược thực tế nếu chưa có gói 0x006a
                if state.bb_size == 0 && state.street == Street::Preflop && max_b > 0 {
                    state.bb_size = max_b;
                    state.sb_size = if second_b > 0 { second_b } else { max_b / 2 };
                    state.min_raise = state.bb_size;
                    println!("💡 [SOCKET PREFLOP] Tự động suy luận Blinds từ cược thực tế: SB = {} chip, BB = {} chip", state.sb_size, state.bb_size);
                }

                println!("💰 [TIỀN CƯỢC THEO GHẾ]: Highest Bet = {} chip", max_b);
            }
        }
        0x0076 => { // CẬP NHẬT STACK (STACK UPDATE)
            let seat = thrift_data.get_field(2).and_then(|v| v.get_i32()).unwrap_or(-1) as usize;
            let raw_stack = thrift_data.get_field(3).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
            let stack = raw_stack / 100;
            let pid = format!("seat_{}", seat);
            if let Some(p) = state.players.get_mut(&pid) {
                p.stack = stack;
                if stack == 0 && (p.current_bet > 0 || p.total_contributed > 0) {
                    p.is_all_in = true;
                }
            }
        }
        0x0077 => { // CẬP NHẬT TIỀN CƯỢC (BET UPDATE)
            let seat = thrift_data.get_field(2).and_then(|v| v.get_i32()).unwrap_or(-1) as usize;
            let raw_bet = thrift_data.get_field(3).and_then(|v| v.get_i32()).unwrap_or(0).max(0) as u32;
            let cur_bet = raw_bet / 100;
            let pid = format!("seat_{}", seat);
            if let Some(p) = state.players.get_mut(&pid) {
                p.current_bet = cur_bet;
            }
            if cur_bet > state.highest_bet {
                let raise_diff = cur_bet - state.highest_bet;
                if raise_diff >= state.min_raise {
                    state.min_raise = raise_diff;
                }
                state.highest_bet = cur_bet;
            }
        }
        0x0071 => { // BÀI CHUNG (COMMUNITY CARDS)
            let cards_list = thrift_data.get_field(2).and_then(|v| v.get_list());
            if let Some(cards) = cards_list {
                let mut parsed_cards = Vec::new();
                for c in cards {
                    let suit = c.get_field(2).and_then(|v| v.get_i32()).unwrap_or(-1);
                    let rank = c.get_field(3).and_then(|v| v.get_i32()).unwrap_or(-1);
                    if let Some(card) = parse_pokerstars_card(suit, rank) {
                        parsed_cards.push(card);
                    }
                }
                if parsed_cards.len() == 3 {
                    // Flop: đúng 3 lá bài mới
                    state.community_cards = parsed_cards;
                    state.street = Street::Flop;
                } else if parsed_cards.len() == 1 {
                    // Turn hoặc River gửi 1 lá mới nối tiếp
                    if state.community_cards.len() == 3 {
                        state.community_cards.extend(parsed_cards);
                        state.street = Street::Turn;
                    } else if state.community_cards.len() >= 4 {
                        state.community_cards.extend(parsed_cards);
                        state.street = Street::River;
                    } else {
                        state.community_cards.extend(parsed_cards);
                    }
                } else if parsed_cards.is_empty() {
                    state.community_cards.clear();
                    state.street = Street::Preflop;
                } else {
                    state.community_cards = parsed_cards;
                    state.street = match state.community_cards.len() {
                        3 => Street::Flop,
                        4 => Street::Turn,
                        5 => Street::River,
                        _ => Street::Preflop,
                    };
                }
                state.has_evaluated_this_decision = false;
                state.highest_bet = 0;
                state.min_raise = state.bb_size;
                for p in state.players.values_mut() {
                    p.current_bet = 0;
                }
                let card_str = state.community_cards.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
                println!("🎴 [BÀI CHUNG ({:?})]: [ {} ]", state.street, card_str);
            } else {
                state.community_cards.clear();
                state.street = Street::Preflop;
                state.has_evaluated_this_decision = false;
                state.highest_bet = 0;
                state.min_raise = state.bb_size;
                for p in state.players.values_mut() {
                    p.current_bet = 0;
                }
                println!("🎴 [BÀI CHUNG (Preflop)]: [ ]");
            }
        }
        0x008b => { // BÀI TẨY HERO TỪ SERVER (HERO HOLE CARDS & HAND EVALUATION)
            let seat = thrift_data.get_field(2).and_then(|v| v.get_i32()).unwrap_or(-1) as usize;
            state.hero_seat = Some(seat);
            let pid = format!("seat_{}", seat);
            if let Some(p) = state.players.get(&pid) {
                if !p.name.is_empty() && p.name != "Player" && state.hero_id.is_empty() {
                    state.hero_id = p.name.clone();
                }
            }
            if let Some(cards_list) = thrift_data.get_field(3).and_then(|v| v.get_list()) {
                let mut hero_cards = Vec::new();
                for c in cards_list {
                    if let (Some(s_val), Some(r_val)) = (c.get_field(2), c.get_field(3)) {
                        if let (Some(s), Some(r)) = (s_val.get_i32(), r_val.get_i32()) {
                            if let Some(card) = parse_pokerstars_card(s, r) {
                                hero_cards.push(card);
                            }
                        }
                    }
                }
                if hero_cards.len() == 2 {
                    let hole = [hero_cards[0], hero_cards[1]];
                    let is_new_cards = state.hero_cards != Some(hole);

                    if is_new_cards && state.community_cards.is_empty() {
                        state.has_evaluated_this_decision = false;
                        state.community_cards.clear();
                        state.action_history.clear();
                        state.street = Street::Preflop;
                        state.highest_bet = 0;
                        state.min_raise = state.bb_size;
                    }

                    state.hero_cards = Some(hole);
                    let hand_desc = thrift_data.get_field(4).and_then(|v| v.get_str()).unwrap_or("");
                    if !hand_desc.is_empty() {
                        println!("🃏 [BÀI TẨY HERO TỪ GÓI 0x008b]: [ {}  {} ] (Ghế #{}) - {}", hole[0], hole[1], seat, hand_desc);
                    } else {
                        println!("🃏 [BÀI TẨY HERO TỪ GÓI 0x008b]: [ {}  {} ] (Ghế #{})", hole[0], hole[1], seat);
                    }

                    // Nếu hiện tại đang là lượt của Hero mà chưa đánh giá EV:
                    if state.current_turn_seat == Some(seat) && !state.has_evaluated_this_decision && state.bb_size > 0 {
                        state.has_evaluated_this_decision = true;
                        let pid = format!("seat_{}", seat);
                        if let Some(p) = state.players.get_mut(&pid) {
                            p.is_folded = false;
                        }
                        println!("🎯🎯🎯 [LƯỢT CỦA HERO - GHẾ #{}] Đã nhận bài tẩy -> Tiến hành tính toán EV ngay! 🎯🎯🎯", seat);
                        return run_gto_evaluation(state, hole, policy, decision_mode, use_os_enigo);
                    }
                }
            }
        }
        0x006f => { // LƯỢT ĐÁNH & ĐỒNG HỒ (ACTION TURN)
            let seat = thrift_data.get_field(2).and_then(|v| v.get_i32()).unwrap_or(-1) as usize;
            let timer = thrift_data.get_field(3).and_then(|v| v.get_i32()).unwrap_or(0);
            println!("👉 [LƯỢT HÀNH ĐỘNG]: Ghế #{} (Thời gian: {:.1}s)", seat, timer as f64 / 1000.0);
            state.current_turn_seat = Some(seat);

            let is_hero_turn = Some(seat) == state.hero_seat 
                || (!state.hero_id.is_empty() && state.players.get(&format!("seat_{}", seat)).map(|p| p.name.eq_ignore_ascii_case(&state.hero_id)).unwrap_or(false));

            if is_hero_turn {
                state.hero_seat = Some(seat);
                let pid = format!("seat_{}", seat);
                if let Some(p) = state.players.get_mut(&pid) {
                    p.is_folded = false;
                }
                println!("🎯🎯🎯 [LƯỢT CỦA HERO - GHẾ #{}] 🎯🎯🎯", seat);
                if !state.has_evaluated_this_decision {
                    // 1. Kiểm tra Blinds từ socket (KHÔNG HARDCODE)
                    if state.bb_size == 0 {
                        println!("⏳ [CHỜ DỮ LIỆU SOCKET] Chưa nhận được thông tin Blinds (BB/SB) từ socket -> Đang chờ socket cập nhật...");
                        return None;
                    }

                    // 2. Kiểm tra bài tẩy Hero từ socket (KHÔNG DÙNG BÀI MẪU HARDCODE)
                    let hole = match state.hero_cards {
                        Some(h) => h,
                        None => {
                            println!("⏳ [CHỜ DỮ LIỆU SOCKET] Chưa nhận diện được 2 lá bài tẩy từ socket -> Chờ socket/DOM gửi bài tẩy thực tế...");
                            return None;
                        }
                    };

                    state.has_evaluated_this_decision = true;
                    return run_gto_evaluation(state, hole, policy, decision_mode, use_os_enigo);
                } else {
                    println!("⏸️ [LƯỢT HERO] Đã đánh giá quyết định cho lượt này rồi (chờ thực thi lệnh cược).");
                }
            } else {
                state.has_evaluated_this_decision = false;
            }
        }
        _ => {}
    }
    None
}

// =========================================================================
// 6. MAIN SERVER & APPLICATION ENTRY POINT (PORT 3002)
// =========================================================================

fn main() {
    println!("============================================================");
    println!("⚡ POKERSTARS REAL-TIME ASSISTANT & AUTO-BOT [THUẦN EV MAX] (PORT 3002) 🚀");
    println!("============================================================");

    print!("📐 Nạp bảng Preflop Equity... ");
    std::io::stdout().flush().unwrap();
    let _ = poker_tournament::equity::preflop_table();
    println!("✅ Xong!");

    let (adv_path, strat_path, is_cash) = if std::path::Path::new(CASH_ADV_MODEL_PATH).exists() {
        (CASH_ADV_MODEL_PATH, CASH_STRAT_MODEL_PATH, true)
    } else {
        (TOUR_ADV_MODEL_PATH, TOUR_STRAT_MODEL_PATH, false)
    };
    IS_CASH_GAME.store(is_cash, Ordering::SeqCst);

    print!("🧠 Nạp mô hình AI [{}]... ", adv_path);
    std::io::stdout().flush().unwrap();
    let adv_mlp = MLP::load_from_file(adv_path).unwrap_or_else(|_| panic!("Lỗi nạp {}", adv_path));
    let strat_mlp = MLP::load_from_file(strat_path).ok();
    println!("✅ Xong! (Chế độ: {}, Strategy Net: {})", 
        if is_cash { "Cash Game Deep-Stack (0..300 BB)" } else { "Tournament SNG" },
        if strat_mlp.is_some() { "Có" } else { "Không" }
    );

    let policy = DeepCFRPolicy::with_strategy(Arc::new(RwLock::new(adv_mlp)), strat_mlp, true);
    let decision_mode = DecisionMode::PureMaxEv;
    let use_os_enigo = std::env::args().any(|a| a == "--enigo" || a == "--use-enigo");

    let listener = TcpListener::bind("127.0.0.1:3002").expect("Không thể mở cổng 3002!");
    println!("🌐 Server sẵn sàng tại http://127.0.0.1:3002");
    println!("🔇 Chế độ thực thi: {}", if use_os_enigo { "🖥️ OS Keyboard (Enigo)" } else { "🥷 100% SILENT IN-PAGE (Tampermonkey DOM)" });
    println!("👀 Đang theo dõi, đồng bộ PokerStars Web và bật Auto-Bot...\n");

    let mut state = HandState::new();
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("pokerstars_dump.jsonl")
        .ok();

    for mut stream in listener.incoming().flatten() {
        let mut raw_data = Vec::with_capacity(32768);
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

                    let mut pending_action: Option<InPageAction> = None;

                    if !trimmed.is_empty() {
                        if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                            // Tự động ghi lại gói tin vào pokerstars_dump.jsonl để phục vụ nghiên cứu
                            if let Some(ref mut f) = log_file {
                                let record = serde_json::json!({
                                    "timestamp_ms": get_timestamp_ms(),
                                    "payload": val
                                });
                                if let Ok(line) = serde_json::to_string(&record) {
                                    let _ = writeln!(f, "{}", line);
                                    let _ = f.flush();
                                }
                            }
                            // 1. Tự động nhận diện Hero ID từ mọi gói WebSocket (STOMP, JSON, Notifications)
                            if let Some(uid) = extract_user_id_from_payload(&val) {
                                if state.hero_id != uid {
                                    state.hero_id = uid.clone();
                                    println!("👤 [SOCKET/WEBSOCKET] Đã tự động nhận diện Hero ID: [{}]", state.hero_id);

                                    // Nếu đã có danh sách ghế, tìm ngay ghế của Hero
                                    if state.hero_seat.is_none() {
                                        for p in state.players.values() {
                                            if p.name.eq_ignore_ascii_case(&state.hero_id) {
                                                state.hero_seat = Some(p.seat_num);
                                                println!("🎯 [HERO NHẬN DIỆN]: Ghế #{} là Hero [{}]!", p.seat_num, p.name);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            // 2. Nhận diện Blinds từ DOM (nếu có gửi kèm)
                            if let Some(bb_val) = val.get("bb").and_then(|v| v.as_u64()) {
                                if state.bb_size == 0 && bb_val > 0 {
                                    state.bb_size = bb_val as u32;
                                    state.sb_size = val.get("sb").and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(state.bb_size / 2);
                                    state.min_raise = state.bb_size;
                                    println!("🏛️ [DOM BLINDS] Đã cập nhật Blinds từ giao diện bàn: SB = {} chip, BB = {} chip", state.sb_size, state.bb_size);
                                }
                            }

                            // 3. Gói tin từ Tampermonkey Sniffer 3 (PokerStars)
                            if val.get("source").and_then(|v| v.as_str()) == Some("pokerstars_web") {
                                if let Some(format) = val.get("format").and_then(|v| v.as_str()) {
                                    if format == "binary_hex" {
                                        if let Some(hex_str) = val.get("data").and_then(|v| v.as_str()) {
                                            let bytes: Vec<u8> = (0..hex_str.len())
                                                .step_by(2)
                                                .filter_map(|i| {
                                                    if i + 2 <= hex_str.len() {
                                                        u8::from_str_radix(&hex_str[i..i+2], 16).ok()
                                                    } else { None }
                                                })
                                                .collect();

                                            if bytes.len() >= 2 {
                                                let opcode = u16::from_be_bytes([bytes[0], bytes[1]]);
                                                if let Some((thrift_data, _)) = parse_thrift_val(12, &bytes[2..], 0) {
                                                    pending_action = process_pokerstars_thrift_packet(
                                                        &mut state, opcode, &thrift_data, &policy, decision_mode, use_os_enigo
                                                    );
                                                }
                                            }
                                        }
                                    }
                                } else if let Some(cards_val) = val.get("cards") {
                                    if let Some(cards) = parse_cards_array(cards_val) {
                                        state.hero_cards = Some(cards);
                                        println!("🃏 [DOM/SOCKET CARDS] Đã nhận diện 2 lá bài tẩy Hero: [ {}  {} ]", cards[0], cards[1]);
                                        if let Some(h_seat) = state.hero_seat {
                                            if state.current_turn_seat == Some(h_seat) && !state.has_evaluated_this_decision && state.bb_size > 0 {
                                                state.has_evaluated_this_decision = true;
                                                println!("🎯🎯🎯 [LƯỢT CỦA HERO - GHẾ #{}] Đã nhận bài tẩy từ DOM -> Tiến hành tính toán EV ngay! 🎯🎯🎯", h_seat);
                                                pending_action = run_gto_evaluation(&state, cards, &policy, decision_mode, use_os_enigo);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let mut resp_map = serde_json::Map::new();
                    resp_map.insert("status".to_string(), serde_json::Value::String("ok".to_string()));
                    if let Some(ref cmd) = pending_action {
                        if let Ok(v) = serde_json::to_value(cmd) {
                            resp_map.insert("bot_action".to_string(), v);
                        }
                    }

                    let resp_body = serde_json::to_string(&serde_json::Value::Object(resp_map))
                        .unwrap_or_else(|_| r#"{"status":"ok"}"#.to_string());
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                        resp_body.len(),
                        resp_body
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    #[test]
    fn test_replay_pokerstars_dump() {
        let dump_path = "pokerstars_dump.jsonl";
        if !std::path::Path::new(dump_path).exists() {
            return;
        }

        let file = File::open(dump_path).expect("Cannot open dump");
        let reader = BufReader::new(file);

        let (adv_path, strat_path, is_cash) = if std::path::Path::new(CASH_ADV_MODEL_PATH).exists() {
            (CASH_ADV_MODEL_PATH, CASH_STRAT_MODEL_PATH, true)
        } else {
            (TOUR_ADV_MODEL_PATH, TOUR_STRAT_MODEL_PATH, false)
        };
        IS_CASH_GAME.store(is_cash, Ordering::SeqCst);
        let adv_mlp = MLP::load_from_file(adv_path).unwrap_or_else(|_| panic!("Lỗi nạp {}", adv_path));
        let strat_mlp = MLP::load_from_file(strat_path).ok();
        let policy = DeepCFRPolicy::with_strategy(Arc::new(RwLock::new(adv_mlp)), strat_mlp, true);
        let mut state = HandState::new();
        let mut hero_turn_evaluated = false;

        for line in reader.lines() {
            let line = line.unwrap();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                let payload = val.get("payload").unwrap_or(&val);
                if payload.get("format").and_then(|v| v.as_str()) == Some("binary_hex") {
                    if let Some(hex_str) = payload.get("data").and_then(|v| v.as_str()) {
                        let bytes: Vec<u8> = (0..hex_str.len())
                            .step_by(2)
                            .filter_map(|i| {
                                if i + 2 <= hex_str.len() {
                                    u8::from_str_radix(&hex_str[i..i+2], 16).ok()
                                } else { None }
                            })
                            .collect();

                        if bytes.len() >= 2 {
                            let opcode = u16::from_be_bytes([bytes[0], bytes[1]]);
                            if let Some((thrift_data, _)) = parse_thrift_val(12, &bytes[2..], 0) {
                                let action = process_pokerstars_thrift_packet(
                                    &mut state, opcode, &thrift_data, &policy, DecisionMode::PureMaxEv, false
                                );
                                if action.is_some() || state.has_evaluated_this_decision {
                                    hero_turn_evaluated = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Verify Hero cards were captured!
        assert!(state.hero_cards.is_some(), "Hero cards should be parsed from 0x008b");
        let cards = state.hero_cards.unwrap();
        println!("Hero cards parsed in test: [ {} {} ]", cards[0], cards[1]);
        assert!(hero_turn_evaluated, "Hero turn EV evaluation should have triggered!");
    }
}

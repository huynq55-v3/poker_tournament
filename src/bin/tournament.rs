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

use enigo::{Enigo, Key, Keyboard, Settings};
use rand::Rng;

const ADV_MODEL_PATH: &str = "deep_cfr_model.json";
const STRAT_MODEL_PATH: &str = "deep_cfr_strategy.json";

// =========================================================================
// 1. DATA STRUCTURES & TYPES
// =========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecisionMode {
    PureMaxEv,     // Mặc định cho receiver2: 100% Argmax EV (Thuần EV Max)
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
}

impl Default for PlayerTracker {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "Player".to_string(),
            seat_num: 99,
            stack: 1000,
            current_bet: 0,
            total_contributed: 0,
            is_folded: false,
            is_all_in: false,
        }
    }
}

#[derive(Debug)]
pub struct HandState {
    pub hand_id: String,
    pub hero_id: String,
    pub hero_cards: Option<[Card; 2]>,
    pub community_cards: Vec<Card>,
    pub street: Street,
    pub pn_total_pot: u32,
    pub highest_bet: u32,
    pub min_raise: u32,
    pub bb_size: u32,
    pub sb_size: u32,
    pub dealer_id: Option<String>,
    pub dealer_seat: Option<usize>,
    pub action_history: Vec<HandActionRecord>,
    pub players: HashMap<String, PlayerTracker>,
    pub seats_order: Vec<String>,
    pub has_evaluated_this_decision: bool,
    pub hero_mavtb: Option<u32>,
}

impl HandState {
    pub fn new() -> Self {
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
            hero_mavtb: None,
        }
    }

    pub fn start_new_hand(&mut self, new_h_id: &str) {
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
        self.hero_mavtb = None;

        // Giữ lại những player ID đang thực sự có mặt ở bàn này
        if !self.seats_order.is_empty() {
            self.players.retain(|id, _| self.seats_order.contains(id));
        }

        for p in self.players.values_mut() {
            p.current_bet = 0;
            p.total_contributed = 0;
            p.is_folded = false;
            p.is_all_in = false;
        }
    }
}

// =========================================================================
// 2. CARD PARSING UTILITIES
// =========================================================================

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
// 3. WINDOW FOCUS & OS AUTOMATION (ENIGO / X11 / WIN32)
// =========================================================================

static TARGET_GAME_ID: RwLock<String> = RwLock::new(String::new());

pub fn set_target_game(url_or_id: &str) {
    let id = if let Some(idx) = url_or_id.rfind("/games/") {
        &url_or_id[idx + 7..]
    } else {
        url_or_id
    };
    let clean_id = id.trim_matches('/').to_string();
    if let Ok(mut lock) = TARGET_GAME_ID.write() {
        *lock = clean_id;
    }
}

pub fn get_target_game_keywords() -> Vec<String> {
    let mut kw = vec![
        "Poker Now".to_string(),
        "pokernow".to_string(),
    ];
    if let Ok(lock) = TARGET_GAME_ID.read() {
        if !lock.is_empty() {
            kw.insert(0, lock.clone());
        }
    }
    kw
}

#[cfg(target_os = "windows")]
mod win_focus {
    use std::ffi::c_void;

    type HWND = *mut c_void;
    type BOOL = i32;
    type DWORD = u32;
    type LPARAM = isize;
    type WNDENUMPROC = Option<unsafe extern "system" fn(HWND, LPARAM) -> BOOL>;

    const SW_RESTORE: i32 = 9;
    const SW_SHOW: i32 = 5;

    #[link(name = "user32")]
    extern "system" {
        fn EnumWindows(lpEnumFunc: WNDENUMPROC, lParam: LPARAM) -> BOOL;
        fn IsWindowVisible(hWnd: HWND) -> BOOL;
        fn GetWindowTextW(hWnd: HWND, lpString: *mut u16, nMaxCount: i32) -> i32;
        fn GetForegroundWindow() -> HWND;
        fn GetWindowThreadProcessId(hWnd: HWND, lpdwProcessId: *mut DWORD) -> DWORD;
        fn AttachThreadInput(idAttach: DWORD, idAttachTo: DWORD, fAttach: BOOL) -> BOOL;
        fn ShowWindow(hWnd: HWND, nCmdShow: i32) -> BOOL;
        fn SetForegroundWindow(hWnd: HWND) -> BOOL;
        fn BringWindowToTop(hWnd: HWND) -> BOOL;
        fn SetFocus(hWnd: HWND) -> HWND;
        fn IsIconic(hWnd: HWND) -> BOOL;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThreadId() -> DWORD;
    }

    struct SearchContext {
        keywords: Vec<String>,
        target_hwnd: Option<HWND>,
    }

    unsafe extern "system" fn enum_window_callback(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let ctx = &mut *(lparam as *mut SearchContext);
        if IsWindowVisible(hwnd) == 0 {
            return 1;
        }

        let mut buffer = [0u16; 512];
        let len = GetWindowTextW(hwnd, buffer.as_mut_ptr(), 512);
        if len > 0 {
            let title = String::from_utf16_lossy(&buffer[..len as usize]);
            for kw in &ctx.keywords {
                if title.contains(kw) {
                    ctx.target_hwnd = Some(hwnd);
                    return 0; // Dừng quét ngay khi tìm thấy
                }
            }
        }
        1
    }

    pub fn focus_pokernow_window() -> bool {
        let mut ctx = SearchContext {
            keywords: super::get_target_game_keywords(),
            target_hwnd: None,
        };

        unsafe {
            EnumWindows(Some(enum_window_callback), &mut ctx as *mut _ as LPARAM);
        }

        if let Some(target) = ctx.target_hwnd {
            unsafe {
                let fg = GetForegroundWindow();
                if fg == target {
                    return true;
                }

                let current_thread = GetCurrentThreadId();
                let fg_thread = GetWindowThreadProcessId(fg, std::ptr::null_mut());

                if fg_thread != 0 && fg_thread != current_thread {
                    AttachThreadInput(current_thread, fg_thread, 1);
                }

                if IsIconic(target) != 0 {
                    ShowWindow(target, SW_RESTORE);
                } else {
                    ShowWindow(target, SW_SHOW);
                }

                BringWindowToTop(target);
                SetForegroundWindow(target);
                SetFocus(target);

                if fg_thread != 0 && fg_thread != current_thread {
                    AttachThreadInput(current_thread, fg_thread, 0);
                }
            }
            true
        } else {
            false
        }
    }
}

#[cfg(target_os = "linux")]
mod linux_focus {
    use std::process::Command;

    pub fn focus_pokernow_window() -> bool {
        let keywords = super::get_target_game_keywords();

        for kw in &keywords {
            let output = Command::new("xdotool")
                .args(["search", "--name", kw])
                .output();

            if let Ok(out) = output {
                if out.status.success() {
                    let text = String::from_utf8_lossy(&out.stdout);
                    if let Some(first_id) = text.lines().next() {
                        let wid = first_id.trim();
                        if !wid.is_empty() {
                            let _ = Command::new("xdotool")
                                .args(["windowactivate", "--sync", wid])
                                .output();
                            return true;
                        }
                    }
                }
            }
        }

        for kw in &keywords {
            let _ = Command::new("wmctrl")
                .args(["-a", kw])
                .output();
        }

        false
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
mod dummy_focus {
    pub fn focus_pokernow_window() -> bool { false }
}

pub fn focus_pokernow_window() -> bool {
    #[cfg(target_os = "windows")]
    { win_focus::focus_pokernow_window() }

    #[cfg(target_os = "linux")]
    { linux_focus::focus_pokernow_window() }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    { dummy_focus::focus_pokernow_window() }
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
            println!("🤖 [BOT] Thao tác: FOLD (Bấm F)");
            let _ = enigo.key(Key::Unicode('f'), enigo::Direction::Click);
        }
        Action::CheckCall => {
            if to_call == 0 {
                println!("🤖 [BOT] Thao tác: CHECK (Bấm K)");
                let _ = enigo.key(Key::Unicode('k'), enigo::Direction::Click);
            } else {
                println!("🤖 [BOT] Thao tác: CALL {} chip (Bấm C)", to_call);
                let _ = enigo.key(Key::Unicode('c'), enigo::Direction::Click);
            }
        }
        Action::AllIn => {
            println!("🤖 [BOT] Thao tác: ALL-IN {} chip (Bấm R, xóa số cũ, nhập số, Enter)", max_total_bet);
            type_bet_amount(&mut enigo, max_total_bet);
        }
        Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
            let final_bet = target_total.min(max_total_bet);
            if final_bet >= max_total_bet {
                println!("🤖 [BOT] Đạt mức trần chip, đổi thành ALL-IN {} chip", max_total_bet);
                type_bet_amount(&mut enigo, max_total_bet);
            } else {
                println!("🤖 [BOT] Thao tác: RAISE lên {} chip (Bấm R, xóa số cũ, nhập số, Enter)", final_bet);
                type_bet_amount(&mut enigo, final_bet);
            }
        }
    }
}

// =========================================================================
// 4. GAME UPDATE PROCESSING & STATE TRACKING
// =========================================================================

pub fn process_game_update(
    state: &mut HandState, 
    payload: &Value, 
    policy: &DeepCFRPolicy, 
    mode: DecisionMode, 
    use_os_enigo: bool
) -> Option<InPageAction> {
    // 1. Nhận diện Hand ID mới
    if let Some(hi) = payload.get("hI").and_then(|v| v.as_str()) {
        if state.hand_id != hi { state.start_new_hand(hi); }
    }

    // 2. Cập nhật Street (Preflop, Flop, Turn, River, Showdown)
    if let Some(gt_arr) = payload.get("gT").and_then(|v| v.as_array()) {
        if gt_arr.len() >= 2 {
            if let Some(st_num) = gt_arr[1].as_u64() {
                let new_street = match st_num {
                    0 => Street::Preflop, 1 => Street::Flop, 2 => Street::Turn,
                    3 => Street::River, _ => Street::Showdown,
                };
                if new_street != state.street {
                    for p in state.players.values_mut() {
                        if p.is_all_in {
                            p.stack = 0;
                        } else {
                            p.stack = p.stack.saturating_sub(p.current_bet);
                        }
                        p.current_bet = 0;
                    }
                    state.highest_bet = 0;
                    state.street = new_street;
                    state.has_evaluated_this_decision = false;
                    state.hero_mavtb = None;
                }
            }
        }
    }

    // 3. Cập nhật danh sách ghế ngồi (Seats) & gán seat_num chuẩn xác
    if let Some(arr) = payload.get("seats").and_then(|v| v.as_array()) {
        let mut v: Vec<(usize, String)> = Vec::new();
        for s in arr {
            let (seat_num, pid_opt) = if let Some(a) = s.as_array() {
                let seat = a.first().and_then(|v| v.as_u64()).map(|n| n as usize);
                let pid = a.get(1).and_then(|v| v.as_str())
                    .or_else(|| a.get(1).and_then(|v| v.get("id")).and_then(|v| v.as_str()))
                    .map(|s| s.to_string());
                (seat, pid)
            } else if let Some(obj) = s.as_object() {
                let seat = obj.get("seat").or_else(|| obj.get("seatNum")).and_then(|v| v.as_u64()).map(|n| n as usize);
                let pid = obj.get("id").or_else(|| obj.get("player_id")).or_else(|| obj.get("playerId")).and_then(|v| v.as_str()).map(|s| s.to_string());
                (seat, pid)
            } else {
                (None, None)
            };

            if let (Some(seat), Some(pid)) = (seat_num, pid_opt) {
                let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                    id: pid.clone(),
                    ..Default::default()
                });
                entry.seat_num = seat;
                v.push((seat, pid));
            }
        }
        if !v.is_empty() {
            v.sort_by_key(|(s, _)| *s);
            state.seats_order = v.into_iter().map(|(_, id)| id).collect();
        }
    } else if let Some(obj) = payload.get("seats").and_then(|v| v.as_object()) {
        let mut v: Vec<(usize, String)> = Vec::new();
        for (k, val) in obj {
            if let Ok(seat) = k.parse::<usize>() {
                if let Some(pid) = val.as_str().or_else(|| val.get("id").and_then(|v| v.as_str())) {
                    let entry = state.players.entry(pid.to_string()).or_insert_with(|| PlayerTracker {
                        id: pid.to_string(),
                        ..Default::default()
                    });
                    entry.seat_num = seat;
                    v.push((seat, pid.to_string()));
                }
            }
        }
        if !v.is_empty() {
            v.sort_by_key(|(s, _)| *s);
            state.seats_order = v.into_iter().map(|(_, id)| id).collect();
        }
    }

    // 4. Kích cỡ Blinds
    if let Some(bb) = payload.get("bigBlind").and_then(|v| v.as_u64()) { state.bb_size = bb as u32; }
    if let Some(sb) = payload.get("smallBlind").and_then(|v| v.as_u64()) { state.sb_size = sb as u32; }
    
    // 5. Pot bàn chơi
    if let Some(p) = payload.get("pot") {
        if let Some(num) = p.as_u64() {
            state.pn_total_pot = num as u32;
        } else if let Some(arr) = p.as_array() {
            state.pn_total_pot = arr.iter().filter_map(|v| v.as_u64()).sum::<u64>() as u32;
        }
    }

    let mut prev_highest = state.highest_bet;
    if let Some(chb) = payload.get("cHB").and_then(|v| v.as_u64()) { state.highest_bet = chb as u32; }
    
    if let Some(mr) = payload.get("mR").and_then(|v| v.as_u64()) {
        state.min_raise = (mr as u32).saturating_sub(state.highest_bet).max(state.bb_size);
    }

    // 6. Cập nhật stacks và tên người chơi
    if let Some(players_obj) = payload.get("players").and_then(|v| v.as_object()) {
        for (pid, pval) in players_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                id: pid.clone(), ..Default::default()
            });
            if let Some(s) = pval.get("stack").and_then(|v| v.as_u64()) { entry.stack = s as u32; }
            if let Some(n) = pval.get("name").and_then(|v| v.as_str()) { entry.name = n.to_string(); }
            if let Some(seat) = pval.get("seat").or_else(|| pval.get("seatNum")).and_then(|v| v.as_u64()) {
                entry.seat_num = seat as usize;
            }
        }
    }

    // Đồng bộ seat_num từ seats_order nếu entry.seat_num chưa xác định
    for (idx, pid) in state.seats_order.iter().enumerate() {
        if let Some(p) = state.players.get_mut(pid) {
            if p.seat_num == 99 {
                p.seat_num = idx + 1;
            }
        }
    }

    // 7. Cập nhật Dealer
    if let Some(did) = payload.get("dealerID").and_then(|v| v.as_str()) {
        state.dealer_id = Some(did.to_string());
    }
    if let Some(ds) = payload.get("dealerSeat").or_else(|| payload.get("dealer")).and_then(|v| v.as_u64()) {
        state.dealer_seat = Some(ds as usize);
    } else if let Some(ref did) = state.dealer_id {
        if let Some(p) = state.players.get(did) {
            if p.seat_num != 99 {
                state.dealer_seat = Some(p.seat_num);
            }
        }
    }

    // 8. Đọc trạng thái Fold / All-In ("pGS") trước "tB"
    if let Some(pgs_obj) = payload.get("pGS").and_then(|v| v.as_object()) {
        for (pid, sval) in pgs_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker { id: pid.clone(), ..Default::default() });
            if let Some(st) = sval.as_str() {
                if st == "fold" { entry.is_folded = true; } 
                else if st == "allIn" { entry.is_all_in = true; }
            }
        }
    }

    // 9. Cập nhật tiền cược hiện tại ("tB") và ghi nhận hành vi Tố vào Action History
    if let Some(tb_obj) = payload.get("tB").and_then(|v| v.as_object()) {
        for (pid, bval) in tb_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker { id: pid.clone(), ..Default::default() });
            let prev_bet = entry.current_bet;
            if let Some(amt) = bval.as_u64() {
                let new_bet = amt as u32;
                if new_bet > prev_bet {
                    let delta = new_bet - prev_bet;
                    entry.total_contributed += delta;

                    // Nếu cược vượt mức highest_bet trước đó -> Ghi nhận Bet / Raise
                    if new_bet > prev_highest && (state.street != Street::Preflop || new_bet > state.bb_size) {
                        if pid != &state.hero_id {
                            state.has_evaluated_this_decision = false;
                        }
                        let act_type = if entry.is_all_in {
                            Action::AllIn
                        } else if delta <= state.bb_size * 2 && state.street != Street::Preflop {
                            Action::BetSmall
                        } else if (delta as f32) < (state.pn_total_pot.max(state.bb_size * 2) as f32 * 0.50) {
                            Action::BetPot33
                        } else {
                            Action::BetPot75
                        };
                        state.action_history.push(HandActionRecord {
                            player_id: pid.clone(),
                            seat_num: entry.seat_num,
                            street: state.street,
                            action: act_type,
                            amount: delta,
                        });
                        prev_highest = new_bet;
                    }
                }
                entry.current_bet = new_bet;
                if new_bet > state.highest_bet {
                    state.highest_bet = new_bet;
                }
            } else if bval.as_str() == Some("<D>") {
                entry.current_bet = 0;
            }
        }
    }

    // 10. Danh sách người chơi còn trong ván ("iHPI")
    if let Some(ihpi) = payload.get("iHPI").and_then(|v| v.as_array()) {
        let active_set: std::collections::HashSet<String> = ihpi.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
        if !active_set.is_empty() {
            for (pid, p) in state.players.iter_mut() {
                if !active_set.contains(pid) { p.is_folded = true; }
            }
        }
    }

    // 11. Bài chung trên bàn ("oTC") - Sắp xếp khóa để đảm bảo thứ tự Flop -> Turn -> River
    if let Some(otc) = payload.get("oTC").and_then(|v| v.as_object()) {
        let mut sorted_keys: Vec<&String> = otc.keys().collect();
        sorted_keys.sort();
        let mut new_board = Vec::new();
        for k in sorted_keys {
            if let Some(arr) = otc.get(k).and_then(|v| v.as_array()) {
                for item in arr {
                    if let Some(s) = item.as_str() { 
                        if let Some(c) = robust_parse_card(s) { new_board.push(c); } 
                    }
                }
            }
        }
        if !new_board.is_empty() && new_board != state.community_cards {
            println!("🎴 [BÀI CHUNG MỚI]: {:?}", new_board);
            state.community_cards = new_board;
            state.has_evaluated_this_decision = false;
        }
    }

    // 12. Bài tẩy ("pC")
    if let Some(pc_obj) = payload.get("pC").and_then(|v| v.as_object()) {
        for (pid, cdata) in pc_obj {
            let is_showdown = cdata
                .get("cards")
                .and_then(|c| c.as_array())
                .is_some_and(|arr| arr.iter().any(|item| item.get("showing") == Some(&Value::Bool(true))));
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

    // 13. Người chơi đến lượt hành động ("pITT" hoặc "cPI")
    let current_actor = payload.get("pITT").or_else(|| payload.get("cPI")).and_then(|v| v.as_str());

    if let Some(actor) = current_actor {
        if let Some(mavtb) = payload.get("mAVTB").and_then(|v| v.as_u64()) {
            let mavtb_u32 = mavtb as u32;
            if actor == state.hero_id {
                state.hero_mavtb = Some(mavtb_u32);
                if let Some(p) = state.players.get_mut(actor) {
                    p.stack = mavtb_u32;
                }
            }
        }

        if !state.hero_id.is_empty() && actor == state.hero_id {
            if !state.has_evaluated_this_decision {
                if let Some(hole) = state.hero_cards {
                    let action = run_gto_evaluation(state, hole, policy, mode, use_os_enigo);
                    if action.is_some() {
                        state.has_evaluated_this_decision = true;
                    }
                    return action;
                } else {
                    println!("⏳ Đến lượt bạn nhưng ván này chưa nhận diện được bài tẩy!");
                }
            }
        } else {
            state.has_evaluated_this_decision = false;
        }
    }

    None
}

// =========================================================================
// 5. GTO DECISION & NEURAL NETWORK INFERENCE
// =========================================================================

pub fn run_gto_evaluation(
    state: &HandState, 
    hole: [Card; 2], 
    policy: &DeepCFRPolicy, 
    mode: DecisionMode, 
    use_os_enigo: bool
) -> Option<InPageAction> {
    // Sắp xếp người chơi theo số ghế tăng dần theo chiều kim đồng hồ quanh bàn
    let mut sorted_players: Vec<&PlayerTracker> = state.players.values().collect();
    sorted_players.sort_by_key(|p| p.seat_num);

    let mut active_seats: Vec<String> = sorted_players
        .iter()
        .filter(|p| {
            let stack_behind = if p.is_all_in {
                0
            } else {
                p.stack.saturating_sub(p.current_bet)
            };
            p.id == state.hero_id || stack_behind > 0 || p.current_bet > 0 || p.is_all_in
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
    let hero_seat = active_seats.iter().position(|id| id == &state.hero_id).unwrap_or(0);

    let mut actual_highest_bet = state.highest_bet;
    let mut sum_current_bets = 0;

    let mut initial_stacks = Vec::with_capacity(num_players);
    let mut env_players = Vec::with_capacity(num_players);

    for (seat_idx, pid) in active_seats.iter().enumerate() {
        let p_track = state.players.get(pid).cloned().unwrap_or_default();
        
        if p_track.current_bet > actual_highest_bet {
            actual_highest_bet = p_track.current_bet;
        }
        
        let true_stack_behind = if p_track.is_all_in {
            0
        } else {
            p_track.stack.saturating_sub(p_track.current_bet)
        };
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

    // Xác định Button Index với cơ chế fallback nếu dealer_id chưa tới
    let button_idx = if let Some(ref did) = state.dealer_id { 
        active_seats.iter().position(|id| id == did).unwrap_or(0) 
    } else if let Some(d_seat) = state.dealer_seat {
        active_seats.iter().position(|id| state.players.get(id).map(|p| p.seat_num) == Some(d_seat)).unwrap_or(0)
    } else { 
        0 
    };

    let mut env = Environment::new(num_players, &initial_stacks, state.bb_size, state.sb_size, button_idx);

    let true_table_pot = (state.pn_total_pot + sum_current_bets).max(state.bb_size + state.sb_size);
    let safe_min_raise = state.min_raise.max(state.bb_size);
    
    // ĐỒNG BỘ TOÀN DIỆN MÔI TRƯỜNG ĐẦY ĐỦ VÀO ENGINE
    env.state.pot = true_table_pot;
    env.state.current_highest_bet = actual_highest_bet;
    env.state.min_raise = safe_min_raise;
    env.state.community_cards = state.community_cards.clone();
    env.state.street = state.street;
    env.state.current_player_idx = hero_seat;
    env.state.button_idx = button_idx;
    env.state.players = env_players;

    // Ánh xạ chính xác lịch sử cược (action_history) theo player_id sang seat index trong Environment
    env.state.action_history = state.action_history.iter().filter_map(|rec| {
        let idx = active_seats.iter().position(|id| id == &rec.player_id)?;
        Some(ActionRecord {
            player_idx: idx,
            street: rec.street,
            action: rec.action,
            amount: rec.amount,
        })
    }).collect();

    let hero_bet = state.players.get(&state.hero_id).map(|p| p.current_bet).unwrap_or(0);
    let hero_stack_behind = env.state.players[hero_seat].stack; 
    let to_call = actual_highest_bet.saturating_sub(hero_bet);
    let max_total_bet = state.hero_mavtb.unwrap_or(hero_stack_behind + hero_bet);

    // Kiểm tra tính khả thi thực tế trên PokerNow UI
    let has_opp_with_chips = state.players.iter().any(|(pid, p)| {
        let opp_stack_behind = if p.is_all_in {
            0
        } else {
            p.stack.saturating_sub(p.current_bet)
        };
        pid != &state.hero_id && !p.is_folded && !p.is_all_in && opp_stack_behind > 0
    });
    let has_chips_to_raise = hero_stack_behind > to_call;
    let mavtb_allows_raise = state.hero_mavtb.is_none_or(|m| m > actual_highest_bet);
    let can_raise = has_opp_with_chips && has_chips_to_raise && mavtb_allows_raise;

    // Lấy ActionMask thuần túy từ engine tính toán
    let mut mask = env.get_action_mask();

    // Chỉ hạn chế tố nếu đối thủ đã All-In hết hoặc UI PokerNow (mAVTB) không cho phép
    if !can_raise {
        mask.0 &= !((1 << Action::BetSmall.to_index())
            | (1 << Action::BetPot33.to_index())
            | (1 << Action::BetPot75.to_index())
            | (1 << Action::AllIn.to_index()));
    }

    // NẠP VECTO ĐẶC TRƯNG VÀ TÍNH TOÁN BẰNG MẠNG NƠ-RON
    let features = env.state.encode_features(hero_seat);

    // 🛡️ CHỐT CHẶN CHỐNG ALL-IN BÀI RÁC Ở PREFLOP (ANTI-TRASH SHOVE GUARD):
    // Nếu đang ở Preflop và bài tẩy quá yếu, khóa cứng nút All-In để bot tuyệt đối không bao giờ đẩy All-in bài rác!
    let chen_strength = features[166]; // F_HAND_STRENGTH (0..1)
    let eff_stack_bb = env.state.effective_stack_bb(hero_seat);
    if state.street == Street::Preflop {
        // Dưới 8 BB thì bài rác < 0.20 mới khóa, còn từ 8 BB trở lên nếu Chen < 0.28 (như 3-6, 4-7, 2-8) thì cấm All-In!
        let min_shove_chen = if eff_stack_bb < 8.0 { 0.20 } else if eff_stack_bb < 15.0 { 0.26 } else { 0.32 };
        if chen_strength < min_shove_chen && mask.is_valid(Action::AllIn) {
            // Đảm bảo vẫn còn ít nhất Fold hoặc CheckCall hợp lệ trước khi khóa AllIn
            if mask.is_valid(Action::Fold) || mask.is_valid(Action::CheckCall) {
                println!(
                    "🛡️  [ANTI-TRASH SHOVE] Bài tẩy {:?} quá yếu (Chen: {:.2} < {:.2} ở {:.1} BB) -> Khóa nút All-In!",
                    hole, chen_strength, min_shove_chen, eff_stack_bb
                );
                mask.0 &= !(1 << Action::AllIn.to_index());
            }
        }
    } else {
        // 🛡️ CHỐT CHẶN BẢO VỆ POSTFLOP (FLOP / TURN / RIVER):
        // 1. Nếu đối thủ Bet/Raise (to_call > 0) và Hero chỉ có High Card (pure air: cat < 0.15)
        //    và KHÔNG có Flush Draw (d[0] < 0.5) cũng KHÔNG có Straight Draw (d[1] < 0.5):
        //    Khóa toàn bộ BetSmall, BetPot33, BetPot75, AllIn! Không được bluff-raise tự sát với bài rác!
        let hand_cat = features[165];
        let has_flush_draw = features[170] >= 0.5;
        let has_straight_draw = features[171] >= 0.5;
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

        // 2. Chống bluff bom All-In / Bet 75% trong Pot nhiều người (Multiway >= 3):
        if in_hand_count >= 3 && is_pure_air {
            println!(
                "👥 [MULTIWAY AIR GUARD] Pot đông người ({} người) với bài rác -> Khóa BetPot75 & All-In!",
                in_hand_count
            );
            mask.0 &= !((1 << Action::BetPot75.to_index()) | (1 << Action::AllIn.to_index()));
        }
    }

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
    println!("💰 Tổng Pot thực tế: {} chip | Cần Call: {} chip {} | Blind: {}/{} chip", true_table_pot, to_call, call_status, state.sb_size, state.bb_size);

    let eff_bb = env.state.effective_stack_bb(hero_seat);
    let eff_chips = (eff_bb * state.bb_size as f32).round() as u32;
    println!("⚡ Effective Stack : {} chip ({:.1} BB)", eff_chips, eff_bb);

    println!("🏦 [STACK CHIP THỰC TẾ]:");
    for pid in &active_seats {
        if let Some(p) = state.players.get(pid) {
            let is_hero = pid == &state.hero_id;
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
            let status_str = if p.is_all_in {
                if p.current_bet > 0 {
                    format!(" [ALL-IN {} chip]", p.current_bet)
                } else {
                    " [ALL-IN]".to_string()
                }
            } else if p.is_folded {
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
    
    // In Lịch sử cược rõ ràng, phản ánh trung thực số chip cược thực tế
    if state.action_history.is_empty() {
        println!("📜 [LỊCH SỬ CƯỢC]: Chưa có cú Bet/Raise nào (Chỉ có Post Blind hoặc Check)");
    } else {
        println!("📜 [LỊCH SỬ CƯỢC - ACTION HISTORY] ({} hành động):", state.action_history.len());
        for (idx, rec) in state.action_history.iter().enumerate() {
            let p_name = state.players.get(&rec.player_id)
                .map(|p| p.name.as_str())
                .unwrap_or("Unknown");
            let is_hero = rec.player_id == state.hero_id;
            let who = if is_hero { format!("{} (Hero)", p_name) } else { p_name.to_string() };
            let seat = state.players.get(&rec.player_id).map(|p| p.seat_num).unwrap_or(rec.seat_num);
            let act_desc = match rec.action {
                Action::AllIn => format!("All-In (+{} chip)", rec.amount),
                Action::Fold => "Fold".to_string(),
                Action::CheckCall => {
                    if rec.amount == 0 { "Check".to_string() } else { format!("Call (+{} chip)", rec.amount) }
                }
                _ => format!("Tố (+{} chip)", rec.amount),
            };
            println!("   {}. [{:?}] {:<12} (Ghế #{:<2}): {}", 
                idx + 1, rec.street, who, seat, act_desc);
        }
    }

    // In tổng chip đã đóng góp vào Pot theo tên người chơi & thứ tự ghế
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
            let label = if p.id == state.hero_id {
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

    // Lọc và chuẩn hóa xác suất GTO
    let mut clean_probs = [0.0f32; NUM_ACTIONS];
    let mut sum_clean = 0.0f32;
    for (i, &p) in probs.iter().enumerate().take(NUM_ACTIONS) {
        if mask.is_index_valid(i) {
            let val = if p > 0.02 { p } else { 0.0 };
            clean_probs[i] = val;
            sum_clean += val;
        }
    }
    if sum_clean <= 1e-4 {
        for (i, &p) in probs.iter().enumerate().take(NUM_ACTIONS) {
            if mask.is_index_valid(i) {
                clean_probs[i] = p;
                sum_clean += p;
            }
        }
    }
    if sum_clean > 0.0 {
        for p in &mut clean_probs { *p /= sum_clean; }
    }

    // Ra quyết định theo chế độ đã cấu hình
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
            // - Phải đạt tần suất GTO tối thiểu >= 8% (loại bỏ hoàn toàn các cú bluff 1-5% do nhiễu neural net)
            // - Và tần suất GTO >= max_gto_p * 0.30 (đảm bảo là nhánh chiến thuật ngang hàng)
            // Trong các ứng viên hợp lệ về GTO này, chọn action có EV (raw_adv) cao nhất!
            let min_gto_threshold = 0.08f32;
            let mut candidates = Vec::new();
            for (i, &ev) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
                if !mask.is_index_valid(i) { continue; }
                let p = clean_probs[i];
                if p >= min_gto_threshold && p >= max_gto_p * 0.30 {
                    candidates.push((i, ev, p));
                }
            }

            // Fallback an toàn: nếu không có action nào đạt ngưỡng, lấy action có max_gto_p
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
        DecisionMode::PureGtoRoll => {
            let mut rng = rand::thread_rng();
            let idx = if let Ok(dist) = rand::distributions::WeightedIndex::new(clean_probs) {
                use rand::distributions::Distribution;
                dist.sample(&mut rng)
            } else {
                let mut b_idx = 0;
                let mut m_p = -1.0;
                for (i, &p) in clean_probs.iter().enumerate() {
                    if p > m_p {
                        m_p = p;
                        b_idx = i;
                    }
                }
                b_idx
            };
            (idx, format!("🎲 [PURE GTO ROLL] Tung xúc xắc theo % lý thuyết ({:.1}%)", clean_probs[idx] * 100.0))
        }
        DecisionMode::GuardedMixing => {
            let mut rng = rand::thread_rng();

            let mut ev_max = f32::NEG_INFINITY;
            let mut best_ev_idx = 0;
            for (i, &ev) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
                if mask.is_index_valid(i) && ev > ev_max {
                    ev_max = ev;
                    best_ev_idx = i;
                }
            }

            let delta_general = 0.20;
            let delta_allin = 0.08;

            let mut safe_candidates = Vec::new();
            let mut blocked_suicidal = Vec::new();
            let mut second_best_ev = f32::NEG_INFINITY;

            for (i, &ev) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
                if !mask.is_index_valid(i) { continue; }
                let act = Action::from_index(i).unwrap();
                let allowed_delta = if act == Action::AllIn { delta_allin } else { delta_general };

                if ev < ev_max && ev > second_best_ev {
                    second_best_ev = ev;
                }

                if ev >= ev_max - allowed_delta && clean_probs[i] > 0.06 {
                    safe_candidates.push(i);
                } else if clean_probs[i] > 0.06 {
                    blocked_suicidal.push((act, ev_max - ev));
                }
            }

            if ev_max - second_best_ev > 0.35 || safe_candidates.len() <= 1 {
                let note = format!("⚡ [ƯU THẾ ÁP ĐẢO] EV vượt trội rõ ràng (+{:.2} so với {:+.2}) -> Chọn 100%", ev_max, second_best_ev);
                (best_ev_idx, note)
            } else {
                let mut guarded_probs = [0.0f32; NUM_ACTIONS];
                let mut sum_guarded = 0.0f32;
                for &idx in &safe_candidates {
                    let w = clean_probs[idx];
                    guarded_probs[idx] = w;
                    sum_guarded += w;
                }

                let picked = if sum_guarded > 0.0 {
                    for p in &mut guarded_probs { *p /= sum_guarded; }
                    if let Ok(dist) = rand::distributions::WeightedIndex::new(guarded_probs) {
                        use rand::distributions::Distribution;
                        dist.sample(&mut rng)
                    } else {
                        best_ev_idx
                    }
                } else {
                    best_ev_idx
                };

                let cand_names: Vec<String> = safe_candidates.iter()
                    .map(|&i| format!("{}: {:.1}%", Action::from_index(i).unwrap().name(), guarded_probs[i] * 100.0))
                    .collect();
                let blocked_str = if blocked_suicidal.is_empty() {
                    "".to_string()
                } else {
                    let b_names: Vec<String> = blocked_suicidal.iter()
                        .map(|(a, diff)| format!("{}(kém {:.2} EV)", a.name(), diff))
                        .collect();
                    format!(" | 🛡️ Đã khóa: [{}]", b_names.join(", "))
                };

                let note = format!("🛡️ [BẢO VỆ RANGE] Ứng viên an toàn: [{}] -> Roll trúng: {:.1}%{}", cand_names.join(", "), guarded_probs[picked] * 100.0, blocked_str);
                (picked, note)
            }
        }
    };

    let get_action_desc = |act: Action| -> String {
        match act {
            Action::Fold => "0".to_string(),
            Action::CheckCall => {
                if to_call == 0 { "Check".to_string() } 
                else { format!("Call {}", to_call) }
            }
            Action::AllIn => format!("All-In ({})", hero_stack_behind),
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                let frac = match act { 
                    Action::BetSmall => 0.15,
                    Action::BetPot33 => 0.33, 
                    Action::BetPot75 => 0.75, 
                    _ => unreachable!(),
                };
                let pot_after_call = true_table_pot + to_call;
                let raise_by = ((pot_after_call as f32) * frac).round() as u32;
                let target_total = (actual_highest_bet + raise_by.max(safe_min_raise)).min(max_total_bet);
                format!("Tố lên {}", target_total)
            }
        }
    };

    println!("📊 BẢNG ĐÁNH GIÁ NƯỚC ĐI [THUẦN EV MAX]:");
    for (i, &p) in probs.iter().enumerate() {
        let act = Action::from_index(i).unwrap();
        let is_valid = mask.is_index_valid(i);
        let rec = if is_valid && i == chosen_idx { 
            match mode {
                DecisionMode::PureMaxEv => " ⭐ TỐI ĐA EV (ƯU TIÊN GTO)",
                DecisionMode::GuardedMixing => " ⭐ ĐÃ CHỌN (GUARDED)",
                DecisionMode::PureGtoRoll => " ⭐ ĐÃ ROLL TRÚNG",
            }
        } else { 
            "" 
        };
        
        let status = if is_valid { 
            "" 
        } else { 
            let eff_stack_bb = env.state.effective_stack_bb(hero_seat);
            if act == Action::Fold && to_call == 0 {
                " (Khóa: Check miễn phí)"
            } else if eff_stack_bb < 12.0 && act != Action::Fold && act != Action::CheckCall && act != Action::AllIn {
                " (Khóa do Effective Stack < 12BB - Push/Fold Mode)"
            } else if act == Action::BetSmall {
                " (Khóa: Trùng size với Bet 33%)"
            } else {
                " (Bị khóa do luật cược / không đủ chip)"
            }
        };

        let raw_val = raw_adv.get(i).copied().unwrap_or(0.0);
        let chip_desc = get_action_desc(act);

        println!("    {:<14}: EV: {:>+6.2} (Tần suất GTO: {:>5.1}%) | [{:<15}]{}{}", act.name(), raw_val, p * 100.0, chip_desc, rec, status);
    }
    
    if let Some(act) = Action::from_index(chosen_idx) {
        println!("{}", decision_note);
        println!("👉 Quyết định cuối cùng: {:?} (EV: {:>+6.2})", act, raw_adv.get(chosen_idx).copied().unwrap_or(0.0));
    }
    println!("============================================================\n");

    // Đóng gói lệnh hành động gửi sang Browser hoặc gõ Enigo
    if let Some(best_action) = Action::from_index(chosen_idx) {
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
                let raise_by = ((pot_after_call as f32) * frac).round() as u32;
                target_total = (actual_highest_bet + raise_by.max(safe_min_raise)).min(max_total_bet);
            }
            Action::AllIn => {
                target_total = max_total_bet;
            }
            _ => {}
        }

        let safe_action = if !can_raise && best_action != Action::Fold && best_action != Action::CheckCall {
            if to_call == 0 {
                println!("⚠️ [BẢO VỆ] Bàn không thể Tố/All-In -> Tự động chuyển thành CHECK!");
            } else {
                println!("⚠️ [BẢO VỆ] Bàn không thể Tố/All-In -> Tự động chuyển thành CALL!");
            }
            Action::CheckCall
        } else {
            best_action
        };

        let delay_ms: u64 = rand::thread_rng().gen_range(1500..=3000);
        println!(
            "⏳ [HUMAN DELAY] Đã tạo độ trễ ngẫu nhiên: {}ms ({:.2}s) trước khi thực hiện nước đi...",
            delay_ms,
            delay_ms as f64 / 1000.0
        );

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

        if use_os_enigo {
            thread::spawn(move || {
                println!("⏳ [ENIGO DELAY] Đang đợi {:.2}s trước khi gõ phím...", delay_ms as f64 / 1000.0);
                thread::sleep(Duration::from_millis(delay_ms));
                execute_bot_action(safe_action, to_call, target_total, max_total_bet, can_raise);
            });
        } else {
            println!(
                "🥷 [SILENT MODE] Lệnh hành động đã gửi ngầm sang Browser: {:?} (Số chip: {}, Trễ: {:.2}s)",
                in_page_cmd.action.to_uppercase(),
                in_page_cmd.amount,
                delay_ms as f64 / 1000.0
            );
        }

        return Some(in_page_cmd);
    }

    None
}

// =========================================================================
// 6. MAIN SERVER & APPLICATION ENTRY POINT
// =========================================================================

fn main() {
    println!("============================================================");
    println!("⚡ POKERNOW REAL-TIME ASSISTANT & AUTO-BOT [THUẦN EV MAX] (v3)");
    println!("============================================================");

    // Cấu hình URL / ID phòng game để tự động focus
    let default_game = "https://www.pokernow.com/games/pglq3HyXKRV6Klg6v5XNQA0_i";
    let mut target_url = default_game.to_string();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if (args[i] == "--game-url" || args[i] == "--game-id" || args[i] == "-g") && i + 1 < args.len() {
            target_url = args[i + 1].clone();
            i += 1;
        }
        i += 1;
    }
    set_target_game(&target_url);
    println!("🎮 Target Game URL: {}", target_url);
    if let Ok(id) = TARGET_GAME_ID.read() {
        println!("🔑 Target Game ID:  [{}]", *id);
    }

    print!("📐 Nạp bảng Preflop Equity... ");
    std::io::stdout().flush().unwrap();
    let _ = poker_tournament::equity::preflop_table();
    println!("✅ Xong!");

    print!("🧠 Nạp mô hình AI... ");
    std::io::stdout().flush().unwrap();
    let adv_mlp = MLP::load_from_file(ADV_MODEL_PATH).expect("Lỗi nạp deep_cfr_model_tour.json");
    let strat_mlp = MLP::load_from_file(STRAT_MODEL_PATH).ok();
    println!("✅ Xong! (Strategy Net: {})", if strat_mlp.is_some() { "Có" } else { "Không" });

    let policy = DeepCFRPolicy::with_strategy(Arc::new(RwLock::new(adv_mlp)), strat_mlp, true);
    let decision_mode = if std::env::args().any(|a| a == "--roll") {
        DecisionMode::PureGtoRoll
    } else if std::env::args().any(|a| a == "--guarded") {
        DecisionMode::GuardedMixing
    } else {
        DecisionMode::PureMaxEv // Mặc định: Đánh thuần theo EV Max!
    };
    let use_os_enigo = std::env::args().any(|a| a == "--enigo" || a == "--use-enigo");

    let listener = TcpListener::bind("127.0.0.1:3000").expect("Không thể mở cổng 3000");
    println!("🌐 Server sẵn sàng tại http://127.0.0.1:3000");
    let mode_desc = match decision_mode {
        DecisionMode::PureMaxEv => "⚡ THUẦN THEO EV MAX (Argmax EV / Advantage - Chọn 100% nước đi có kỳ vọng cao nhất, Không Roll xúc xắc) 🚀",
        DecisionMode::GuardedMixing => "🛡️ TOP-EV GUARDED MIXING (Bảo vệ range + Chống All-in ngớ ngẩn + Trọng số EV-GTO)",
        DecisionMode::PureGtoRoll => "🎲 PURE GTO ROLL (Tung xúc xắc theo % GTO nguyên bản)",
    };
    println!("🎯 Chế độ ra quyết định: {}", mode_desc);
    println!("🔇 Chế độ thực thi: {}", if use_os_enigo { "🖥️ OS Keyboard (Enigo - Cần cửa sổ focus)" } else { "🥷 100% SILENT IN-PAGE (Tampermonkey DOM - Không chiếm chuột, không chiếm phím, vừa làm việc vừa chơi)" });
    println!("👀 Đang theo dõi, đồng bộ PokerNow và bật Auto-Bot...\n");

    let mut state = HandState::new();

    for mut stream in listener.incoming().flatten() {
        let _ = stream.set_read_timeout(Some(Duration::from_millis(5000)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(5000)));
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

                    let mut pending_action: Option<InPageAction> = None;
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
                                            pending_action = process_game_update(&mut state, gs, &policy, decision_mode, use_os_enigo);
                                        }
                                    } else if event_name == "gC" {
                                        pending_action = process_game_update(&mut state, payload, &policy, decision_mode, use_os_enigo);
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
                    if stream.write_all(response.as_bytes()).is_err() || stream.flush().is_err() {
                        if pending_action.is_some() {
                            println!("⚠️ [MẠNG] Không thể gửi lệnh sang Browser (Client disconnected/timeout) -> Tự động reset cờ để tính lại!");
                            state.has_evaluated_this_decision = false;
                        }
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postflop_anti_suicide_guard_blocks_air_raise() {
        let adv_mlp = MLP::load_from_file(ADV_MODEL_PATH).expect("Cannot load model");
        let strat_mlp = MLP::load_from_file(STRAT_MODEL_PATH).ok();
        let policy = DeepCFRPolicy::with_strategy(Arc::new(RwLock::new(adv_mlp)), strat_mlp, true);

        let mut state = HandState::new();
        state.hand_id = "test_hand".to_string();
        state.hero_id = "hero1".to_string();
        state.street = Street::Flop;
        state.bb_size = 20;
        state.sb_size = 10;
        state.highest_bet = 60;
        state.pn_total_pot = 120;
        state.community_cards = vec![
            robust_parse_card("Ad").unwrap(),
            robust_parse_card("Kd").unwrap(),
            robust_parse_card("Jc").unwrap(),
        ];

        let mut hero = PlayerTracker::default();
        hero.id = "hero1".to_string();
        hero.name = "Hero".to_string();
        hero.stack = 1000;
        hero.current_bet = 0;
        hero.seat_num = 1;

        let mut opp = PlayerTracker::default();
        opp.id = "opp1".to_string();
        opp.name = "Villain".to_string();
        opp.stack = 1000;
        opp.current_bet = 60;
        opp.seat_num = 2;

        state.players.insert(hero.id.clone(), hero);
        state.players.insert(opp.id.clone(), opp);
        state.seats_order = vec!["hero1".to_string(), "opp1".to_string()];

        // Hero holds pure trash: 7c 2s on A-K-J board (High Card, no draws, facing a bet of 60)
        let hole = [robust_parse_card("7c").unwrap(), robust_parse_card("2s").unwrap()];

        let action = run_gto_evaluation(&state, hole, &policy, DecisionMode::PureMaxEv, false);
        assert!(action.is_some(), "Should produce an action");
        let act = action.unwrap();
        println!("Action produced for pure air facing bet: {:?}", act);
        // It must NOT raise or all-in with pure air facing a bet!
        assert!(act.action == "fold" || act.action == "call", "Hero must Fold or Call, never Raise/All-In with pure air! Got: {}", act.action);
    }
}


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

// Thư viện điều khiển bàn phím tự động
use enigo::{Enigo, Key, Keyboard, Settings};

const ADV_MODEL_PATH: &str = "deep_cfr_model.json";
const STRAT_MODEL_PATH: &str = "deep_cfr_strategy.json";
pub const BOT_RESEARCH_NOTICE: &str = "Notice: You are playing against an AI bot for research purposes only.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DecisionMode {
    PureMaxEv,     // Mặc định cho receiver2: 100% Argmax EV (Thuần EV Max)
    GuardedMixing, // --guarded: Lọc phòng hộ EV + Bảo vệ range GTO
    PureGtoRoll,   // --roll: 100% GTO Roll nguyên bản
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct InPageAction {
    pub action: String, // "fold", "check", "call", "all_in", "raise"
    pub amount: u32,
    pub to_call: u32,
}

#[derive(Clone, Debug)]
struct PlayerTracker {
    id: String,
    name: String,
    seat_num: usize,
    stack: u32,
    current_bet: u32,
    total_contributed: u32,
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
            total_contributed: 0,
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
    hero_mavtb: Option<u32>,
    has_announced_chat: bool,
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
            hero_mavtb: None,
            has_announced_chat: false,
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
        self.hero_mavtb = None;

        // 🟢 FIX: Nếu có danh sách ghế ngồi thực tế từ PokerNow (seats_order),
        // chỉ giữ lại những player ID đang thực sự có mặt ở bàn này!
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

fn type_bet_amount(enigo: &mut Enigo, amount: u32) {
    let _ = enigo.key(Key::Unicode('r'), enigo::Direction::Click);
    thread::sleep(Duration::from_millis(100));
    // Xóa sạch số mặc định cũ trong ô cược của PokerNow (tránh bị nối chuỗi số như 401000)
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

/// Tự động gửi tin nhắn chat qua phím 'm', gõ nội dung, và bấm Enter
pub fn send_chat_message_enigo(message: &str) {
    focus_pokernow_window();
    thread::sleep(Duration::from_millis(250));

    let mut enigo = match Enigo::new(&Settings::default()) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("❌ [ENIGO] Không thể khởi tạo Enigo để chat: {:?}", err);
            return;
        }
    };

    println!("💬 [ENIGO] Bấm 'm' mở chat và gửi thông báo: \"{}\"", message);
    let _ = enigo.key(Key::Unicode('m'), enigo::Direction::Click);
    thread::sleep(Duration::from_millis(150));

    for ch in message.chars() {
        let _ = enigo.key(Key::Unicode(ch), enigo::Direction::Click);
        thread::sleep(Duration::from_millis(20));
    }
    thread::sleep(Duration::from_millis(100));

    let _ = enigo.key(Key::Return, enigo::Direction::Click);
    thread::sleep(Duration::from_millis(150));

    // Thoát focus khung chat để không làm kẹt phím bấm poker ('f', 'k', 'c', 'r')
    let _ = enigo.key(Key::Escape, enigo::Direction::Click);
}

// =========================================================================
// 🪟 WINDOW FOCUS MODULE (KHÔNG CHIẾM CHUỘT / KHÔNG DI CHUỘT)
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

        let mut buf = [0u16; 512];
        let len = GetWindowTextW(hwnd, buf.as_mut_ptr(), 512);
        if len > 0 {
            let title = String::from_utf16_lossy(&buf[..len as usize]).to_lowercase();
            for kw in &ctx.keywords {
                if title.contains(&kw.to_lowercase()) {
                    ctx.target_hwnd = Some(hwnd);
                    return 0; // Tìm thấy cửa sổ khớp, dừng duyệt
                }
            }
        }
        1
    }

    pub fn focus_window(keywords: &[String]) -> bool {
        let mut ctx = SearchContext {
            keywords: keywords.to_vec(),
            target_hwnd: None,
        };

        unsafe {
            EnumWindows(Some(enum_window_callback), &mut ctx as *mut _ as LPARAM);

            if let Some(hwnd) = ctx.target_hwnd {
                let fg_wnd = GetForegroundWindow();
                let fg_thread = GetWindowThreadProcessId(fg_wnd, std::ptr::null_mut());
                let cur_thread = GetCurrentThreadId();

                if fg_thread != cur_thread && fg_thread != 0 {
                    AttachThreadInput(cur_thread, fg_thread, 1);
                }

                if IsIconic(hwnd) != 0 {
                    ShowWindow(hwnd, SW_RESTORE);
                } else {
                    ShowWindow(hwnd, SW_SHOW);
                }

                SetForegroundWindow(hwnd);
                BringWindowToTop(hwnd);
                SetFocus(hwnd);

                if fg_thread != cur_thread && fg_thread != 0 {
                    AttachThreadInput(cur_thread, fg_thread, 0);
                }
                true
            } else {
                false
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod linux_focus {
    use std::process::Command;

    pub fn focus_window(keywords: &[String]) -> bool {
        // 1. Thử dùng wmctrl nếu có (chuẩn X11 / Xwayland)
        for kw in keywords {
            if let Ok(status) = Command::new("wmctrl").args(["-a", kw]).status() {
                if status.success() {
                    return true;
                }
            }
        }

        // 2. Thử dùng xdotool nếu có
        for kw in keywords {
            if let Ok(output) = Command::new("xdotool").args(["search", "--name", kw]).output() {
                if output.status.success() {
                    let wids = String::from_utf8_lossy(&output.stdout);
                    if let Some(first_id) = wids.lines().next() {
                        let trimmed = first_id.trim();
                        if !trimmed.is_empty() {
                            let _ = Command::new("xdotool").args(["windowactivate", trimmed]).status();
                            return true;
                        }
                    }
                }
            }
        }

        // 3. GNOME Wayland / Desktop Apps (Kích hoạt cửa sổ trình duyệt đang chạy)
        for app in &["google-chrome", "google-chrome.desktop", "brave-browser", "firefox", "chromium"] {
            if let Ok(status) = Command::new("gtk-launch").arg(app).status() {
                if status.success() {
                    return true;
                }
            }
        }

        false
    }
}

/// Tự động đưa cửa sổ PokerNow lên foreground mà KHÔNG chiếm chuột, KHÔNG di chuyển trỏ chuột
pub fn focus_pokernow_window() {
    let keywords = get_target_game_keywords();

    #[cfg(target_os = "windows")]
    {
        if win_focus::focus_window(&keywords) {
            println!("🪟 [FOCUS] Đã tự động focus cửa sổ PokerNow (Windows - không chiếm chuột)!");
        } else {
            println!("⚠️ [FOCUS] Chưa tìm thấy cửa sổ PokerNow trên Windows.");
        }
    }

    #[cfg(target_os = "linux")]
    {
        if linux_focus::focus_window(&keywords) {
            println!("🪟 [FOCUS] Đã kích hoạt cửa sổ PokerNow trên Linux (không chiếm chuột)!");
        } else {
            println!("⚠️ [FOCUS] Đã gửi lệnh kích hoạt cửa sổ trình duyệt (Linux).");
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        println!("ℹ️ Nền tảng chưa hỗ trợ window focus tự động.");
    }
}

// Hàm thực hiện tự động gõ phím vào cửa sổ đang focus
fn execute_bot_action(action: Action, to_call: u32, target_total: u32, max_total_bet: u32, can_raise: bool) {
    // 🪟 TỰ ĐỘNG FOCUS CỬA SỔ POKERNOW (KHÔNG CHIẾM CHUỘT / KHÔNG DI CHUỘT)
    focus_pokernow_window();

    // Chờ 250ms để cửa sổ nhận focus trọn vẹn và giao diện sẵn sàng nhận phím
    thread::sleep(Duration::from_millis(250));

    let mut enigo = Enigo::new(&Settings::default()).unwrap();

    // 🛡️ BẢO VỆ TUYỆT ĐỐI: Nếu bàn không thể Tố (đối thủ đã All-In hoặc không còn ai có chip),
    // tự động chuyển hướng an toàn: nếu to_call == 0 thì CHECK, nếu to_call > 0 thì CALL!
    let safe_action = if !can_raise && action != Action::Fold && action != Action::CheckCall {
        if to_call == 0 {
            println!("⚠️ [BẢO VỆ] Bàn không thể Tố/All-In -> Tự động chuyển thành CHECK!");
        } else {
            println!("⚠️ [BẢO VỆ] Bàn không thể Tố/All-In -> Tự động chuyển thành CALL!");
        }
        Action::CheckCall
    } else {
        action
    };

    match safe_action {
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
    let adv_mlp = MLP::load_from_file(ADV_MODEL_PATH).expect("Lỗi nạp deep_cfr_model.json");
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
                        let mut chat_to_send: Option<String> = None;
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

                                            // Thông báo bot nghiên cứu khi vào bàn chơi
                                            if !state.has_announced_chat {
                                                state.has_announced_chat = true;
                                                println!("📢 [THÔNG BÁO] Gửi tin nhắn thông báo bot nghiên cứu vào bàn chơi...");
                                                chat_to_send = Some(BOT_RESEARCH_NOTICE.to_string());
                                                if use_os_enigo {
                                                    let notice = BOT_RESEARCH_NOTICE.to_string();
                                                    thread::spawn(move || {
                                                        thread::sleep(Duration::from_millis(800));
                                                        send_chat_message_enigo(&notice);
                                                    });
                                                }
                                            }

                                            if let Some(gs) = payload.get("gameState") {
                                                pending_action = process_game_update(&mut state, gs, &policy, decision_mode, use_os_enigo);
                                            }
                                        } else if event_name == "gC" {
                                            // Phòng khi gói registered bị trôi qua trước khi Rust khởi động
                                            if !state.has_announced_chat && !state.hand_id.is_empty() {
                                                state.has_announced_chat = true;
                                                println!("📢 [THÔNG BÁO] Gửi tin nhắn thông báo bot nghiên cứu vào bàn chơi (vào giữa ván)...");
                                                chat_to_send = Some(BOT_RESEARCH_NOTICE.to_string());
                                                if use_os_enigo {
                                                    let notice = BOT_RESEARCH_NOTICE.to_string();
                                                    thread::spawn(move || {
                                                        thread::sleep(Duration::from_millis(800));
                                                        send_chat_message_enigo(&notice);
                                                    });
                                                }
                                            }

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
                        if let Some(ref chat_msg) = chat_to_send {
                            resp_map.insert("chat_message".to_string(), serde_json::Value::String(chat_msg.clone()));
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

fn process_game_update(state: &mut HandState, payload: &Value, policy: &DeepCFRPolicy, mode: DecisionMode, use_os_enigo: bool) -> Option<InPageAction> {
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
                        // PokerNow cập nhật stack thực tế liên tục qua players.stack và mAVTB,
                        // chỉ reset current_bet về 0 ở vòng mới, không trừ đúp stack!
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

    if let Some(arr) = payload.get("seats").and_then(|v| v.as_array()) {
        let mut v: Vec<(u64, String)> = arr.iter().filter_map(|s| {
            let a = s.as_array()?;
            Some((a.first()?.as_u64()?, a.get(1)?.as_str()?.to_string()))
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

    // 🟢 Lưu lại mức cược cao nhất trước khi đọc "cHB" từ gói JSON mới
    let mut prev_highest = state.highest_bet;

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

    // 🟢 Đọc gói "pGS" (fold / allIn) TRƯỚC "tB" để biết chính xác entry.is_all_in!
    if let Some(pgs_obj) = payload.get("pGS").and_then(|v| v.as_object()) {
        for (pid, sval) in pgs_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker { id: pid.clone(), ..Default::default() });
            if let Some(st) = sval.as_str() {
                if st == "fold" { entry.is_folded = true; } 
                else if st == "allIn" { entry.is_all_in = true; }
            }
        }
    }

    if let Some(tb_obj) = payload.get("tB").and_then(|v| v.as_object()) {
        for (pid, bval) in tb_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker { id: pid.clone(), ..Default::default() });
            let prev_bet = entry.current_bet;
            if let Some(amt) = bval.as_u64() {
                let new_bet = amt as u32;
                if new_bet > prev_bet {
                    let delta = new_bet - prev_bet;
                    entry.total_contributed += delta;

                    // Nếu người chơi này cược vượt mức highest_bet trước đó -> Họ vừa Bet / Raise!
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
                        state.action_history.push(ActionRecord {
                            player_idx: entry.seat_num,
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

    let current_actor = payload.get("pITT").or_else(|| payload.get("cPI")).and_then(|v| v.as_str());

    if let Some(actor) = current_actor {
        if let Some(mavtb) = payload.get("mAVTB").and_then(|v| v.as_u64()) {
            let mavtb_u32 = mavtb as u32;
            if actor == state.hero_id {
                state.hero_mavtb = Some(mavtb_u32);
                if let Some(p) = state.players.get_mut(actor) {
                    // mAVTB là tổng cược tối đa hợp lệ trên PokerNow (bao gồm cả bet hiện tại).
                    // Stack còn lại sau lưng của người chơi = mAVTB - current_bet.
                    p.stack = mavtb_u32.saturating_sub(p.current_bet);
                }
            }
        }

        if !state.hero_id.is_empty() && actor == state.hero_id {
            if !state.has_evaluated_this_decision {
                if let Some(hole) = state.hero_cards {
                    state.has_evaluated_this_decision = true;
                    return run_gto_evaluation(state, hole, policy, mode, use_os_enigo);
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

fn run_gto_evaluation(state: &HandState, hole: [Card; 2], policy: &DeepCFRPolicy, mode: DecisionMode, use_os_enigo: bool) -> Option<InPageAction> {
    let mut sorted_players: Vec<&PlayerTracker> = state.players.values().collect();
    sorted_players.sort_by_key(|p| p.seat_num);

    let active_seats: Vec<String> = sorted_players
        .into_iter()
        .filter(|p| p.id == state.hero_id || p.stack > 0 || p.current_bet > 0 || p.is_all_in)
        .map(|p| p.id.clone())
        .collect();

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
        
        let true_stack_behind = p_track.stack;
        let starting_stack = true_stack_behind + p_track.current_bet;
        initial_stacks.push(starting_stack.max(state.bb_size));

        let mut ps = PlayerState::new(seat_idx, true_stack_behind, seat_idx != hero_seat);
        ps.current_bet = p_track.current_bet;
        ps.total_contributed = p_track.total_contributed.max(p_track.current_bet);
        ps.is_folded = p_track.is_folded;
        ps.is_all_in = p_track.is_all_in;
        if seat_idx == hero_seat { ps.hole_cards = [Some(hole[0]), Some(hole[1])]; }
        
        sum_current_bets += p_track.current_bet;
        env_players.push(ps);
    }

    let in_hand_count = active_seats.iter().filter(|pid| {
        state.players.get(*pid).is_some_and(|pl| !pl.is_folded)
    }).count();

    let button_idx = if let Some(ref did) = state.dealer_id { active_seats.iter().position(|id| id == did).unwrap_or(0) } else { 0 };
    let mut env = Environment::new(num_players, &initial_stacks, state.bb_size, state.sb_size, button_idx);

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
    env.state.action_history = state.action_history.iter().filter_map(|rec| {
        let idx = active_seats.iter().position(|id| {
            state.players.get(id).map(|p| p.seat_num) == Some(rec.player_idx)
        })?;
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

    // 🛡️ Kiểm tra xem trên bàn còn đối thủ nào có chip để tố hay không
    let has_opp_with_chips = state.players.iter().any(|(pid, p)| {
        pid != &state.hero_id && !p.is_folded && !p.is_all_in && p.stack > 0
    });
    let has_chips_to_raise = hero_stack_behind > to_call;
    let mavtb_allows_raise = state.hero_mavtb.is_none_or(|m| m > actual_highest_bet);
    let can_raise = has_opp_with_chips && has_chips_to_raise && mavtb_allows_raise;

    let mut mask = env.get_action_mask();

    // 🛡️ Nếu không thể tố (đối thủ đã All-In hoặc không còn ai có chip), khóa sạch các action tố/All-In
    if !can_raise {
        mask.0 &= !((1 << Action::BetSmall.to_index())
            | (1 << Action::BetPot33.to_index())
            | (1 << Action::BetPot75.to_index())
            | (1 << Action::AllIn.to_index()));
    }

    // 🛡️ BẢO VỆ CHỐNG MIN-BET (ANTI-MINBET / POT-ODDS SHIELD):
    // CHỈ áp dụng ở Postflop (Flop/Turn/River) khi Pot đã to mà đối thủ bet thăm dò 1-2 BB (pot_odds < 12%)!
    // Tuyệt đối KHÔNG áp dụng ở Preflop vì ở Preflop to_call <= 2 BB là bình thường và bot phải được quyền Fold bài rác (như Q7, 72)!
    let pot_odds = if true_table_pot + to_call > 0 {
        to_call as f32 / (true_table_pot + to_call) as f32
    } else {
        0.0
    };

    if state.street != Street::Preflop && to_call > 0 && (to_call <= state.bb_size * 2 || pot_odds < 0.12) {
        println!("🛡️ [POT-ODDS SHIELD] Đối thủ Min-bet {} chip ở {:?} (Pot Odds: {:.1}%) -> Khóa FOLD, ép Call/Raise!", 
            to_call, state.street, pot_odds * 100.0);
        mask.0 &= !(1 << Action::Fold.to_index());
    }

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
    println!("💰 Tổng Pot thực tế: {} chip | Cần Call: {} chip {}", true_table_pot, to_call, call_status);
    println!("------------------------------------------------------------");

    // LỌC VÀ CHUẨN HÓA XÁC SUẤT THEO GTO MIXED STRATEGY
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
                clean_probs[i] = p.max(0.01);
                sum_clean += clean_probs[i];
            }
        }
    }
    if sum_clean > 0.0 {
        for p in &mut clean_probs {
            *p /= sum_clean;
        }
    }

    // CHỌN HÀNH ĐỘNG THEO CHẾ ĐỘ
    let (chosen_idx, decision_note) = match mode {
        DecisionMode::PureMaxEv => {
            // 1. PURE MAX-EV GREEDY (Argmax trên tất cả các action hợp lệ)
            let mut b_idx = 0;
            let mut b_ev = f32::NEG_INFINITY;
            for (i, &adv) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
                if mask.is_index_valid(i)
                    && (adv > b_ev + 1e-4 || ((adv - b_ev).abs() <= 1e-4 && clean_probs[i] > clean_probs[b_idx]))
                {
                    b_ev = adv;
                    b_idx = i;
                }
            }
            let best_act = Action::from_index(b_idx).unwrap();
            (b_idx, format!("⚡ [THUẦN EV MAX] Nước đi tối đa kỳ vọng: {:?} (EV: {:>+6.2})", best_act, b_ev))
        }
        DecisionMode::PureGtoRoll => {
            // 2. PURE GTO RANDOM ROLL (WeightedIndex trên toàn bộ clean_probs)
            let mut rng = rand::thread_rng();
            let idx = if let Ok(dist) = rand::distributions::WeightedIndex::new(clean_probs) {
                use rand::distributions::Distribution;
                dist.sample(&mut rng)
            } else {
                let mut b_idx = 0;
                let mut b_p = -1.0;
                for (i, &p) in clean_probs.iter().enumerate() {
                    if mask.is_index_valid(i) && p > b_p {
                        b_p = p;
                        b_idx = i;
                    }
                }
                b_idx
            };
            (idx, format!("🎲 [PURE GTO ROLL] Tung xúc xắc theo % lý thuyết ({:.1}%)", clean_probs[idx] * 100.0))
        }
        DecisionMode::GuardedMixing => {
            // 3. TOP-EV GUARDED MIXING:
            //   - Phòng hộ: Khóa mọi action tự sát (đặc biệt là All-In rác)
            //   - Vượt trội: Nếu 1 nước chênh > 0.35 BB, chọn thẳng 100%
            //   - Bảo vệ range: Nếu nhiều nước xấp xỉ EV, tung xúc xắc theo % GTO đã tái chuẩn hóa
            let mut rng = rand::thread_rng();

            // Tìm EV cao nhất trong các action hợp lệ
            let mut ev_max = f32::NEG_INFINITY;
            let mut best_ev_idx = 0;
            for (i, &ev) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
                if mask.is_index_valid(i) && ev > ev_max {
                    ev_max = ev;
                    best_ev_idx = i;
                }
            }

            let delta_general = 0.20; // Ngưỡng dung sai EV thông thường (0.20 BB)
            let delta_allin = 0.08;   // Ngưỡng cực kỳ khắt khe cho All-In (0.08 BB) để triệt tiêu All-in ngớ ngẩn

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

                if ev >= ev_max - allowed_delta && clean_probs[i] > 0.02 {
                    safe_candidates.push(i);
                } else if clean_probs[i] > 0.02 {
                    blocked_suicidal.push((act, ev_max - ev));
                }
            }

            // A. Trường hợp ưu thế áp đảo (> 0.35 BB) hoặc chỉ có 1 ứng viên an toàn duy nhất:
            if ev_max - second_best_ev > 0.35 || safe_candidates.len() <= 1 {
                let note = format!("⚡ [ƯU THẾ ÁP ĐẢO] EV vượt trội rõ ràng (+{:.2} so với {:+.2}) -> Chọn 100%", ev_max, second_best_ev);
                (best_ev_idx, note)
            } else {
                // B. Trường hợp các nước đi tương đương EV: Tung xúc xắc bảo vệ range!
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
                else if hero_stack_behind <= to_call { format!("Call All-in ({})", hero_stack_behind) } 
                else { format!("Call {}", to_call) }
            },
            Action::AllIn => format!("All-In ({})", max_total_bet),
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
                DecisionMode::PureMaxEv => " ⭐ TỐI ĐA EV (CHỌN 100%)",
                DecisionMode::GuardedMixing => " ⭐ ĐÃ CHỌN (GUARDED)",
                DecisionMode::PureGtoRoll => " ⭐ ĐÃ ROLL TRÚNG",
            }
        } else { 
            "" 
        };
        
        let status = if is_valid { 
            "" 
        } else { 
            let hero_stack_bb = hero_stack_behind as f32 / state.bb_size as f32;
            if !can_raise && act != Action::Fold && act != Action::CheckCall {
                " (Khóa do đối thủ All-In / không thể tố)"
            } else if hero_stack_bb < 12.0 && act != Action::Fold && act != Action::CheckCall && act != Action::AllIn {
                " (Khóa do Stack < 12BB - Push/Fold Mode)"
            } else if act == Action::BetSmall || act == Action::BetPot75 {
                " (Khóa do trùng size/hết chip)"
            } else {
                " (Bị khóa)"
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

    // TỰ ĐỘNG THỰC THI HÀNH ĐỘNG ĐÃ ROLL ĐƯỢC LÊN GIAO DIỆN
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

        let in_page_cmd = match safe_action {
            Action::Fold => InPageAction {
                action: "fold".to_string(),
                amount: 0,
                to_call: 0,
            },
            Action::CheckCall => {
                if to_call == 0 {
                    InPageAction {
                        action: "check".to_string(),
                        amount: 0,
                        to_call: 0,
                    }
                } else {
                    InPageAction {
                        action: "call".to_string(),
                        amount: to_call,
                        to_call,
                    }
                }
            }
            Action::AllIn => InPageAction {
                action: "all_in".to_string(),
                amount: max_total_bet,
                to_call,
            },
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                let final_bet = target_total.min(max_total_bet);
                if final_bet >= max_total_bet {
                    InPageAction {
                        action: "all_in".to_string(),
                        amount: max_total_bet,
                        to_call,
                    }
                } else {
                    InPageAction {
                        action: "raise".to_string(),
                        amount: final_bet,
                        to_call,
                    }
                }
            }
        };

        if use_os_enigo {
            execute_bot_action(safe_action, to_call, target_total, max_total_bet, can_raise);
        } else {
            println!("🥷 [SILENT MODE] Lệnh hành động đã gửi ngầm sang Browser: {:?} (Số chip: {})", in_page_cmd.action.to_uppercase(), in_page_cmd.amount);
        }

        return Some(in_page_cmd);
    }

    None
}

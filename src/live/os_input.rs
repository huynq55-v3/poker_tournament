use crate::environment::Action;
use enigo::{Enigo, Key, Keyboard, Settings};
use std::sync::RwLock;
use std::thread;
use std::time::Duration;

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
    let mut kw = vec!["Poker Now".to_string(), "pokernow".to_string()];
    if let Ok(lock) = TARGET_GAME_ID.read() {
        if !lock.is_empty() {
            kw.insert(0, lock.clone());
        }
    }
    kw
}

#[cfg(target_os = "windows")]
mod platform_focus {
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
                    return 0;
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
                ShowWindow(target, if IsIconic(target) != 0 { SW_RESTORE } else { SW_SHOW });
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
mod platform_focus {
    use std::process::Command;

    pub fn focus_pokernow_window() -> bool {
        let keywords = super::get_target_game_keywords();
        for kw in &keywords {
            if let Ok(out) = Command::new("xdotool").args(["search", "--name", kw]).output() {
                if out.status.success() {
                    if let Some(first_id) = String::from_utf8_lossy(&out.stdout).lines().next() {
                        let wid = first_id.trim();
                        if !wid.is_empty() {
                            let _ = Command::new("xdotool").args(["windowactivate", "--sync", wid]).output();
                            return true;
                        }
                    }
                }
            }
        }
        for kw in &keywords {
            let _ = Command::new("wmctrl").args(["-a", kw]).output();
        }
        false
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
mod platform_focus {
    pub fn focus_pokernow_window() -> bool {
        false
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
    platform_focus::focus_pokernow_window();
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
        let key = if to_call == 0 { 'k' } else { 'c' };
        let _ = enigo.key(Key::Unicode(key), enigo::Direction::Click);
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
            println!("🤖 [BOT] Thao tác: ALL-IN {} chip", max_total_bet);
            type_bet_amount(&mut enigo, max_total_bet);
        }
        Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
            let final_bet = target_total.min(max_total_bet);
            if final_bet >= max_total_bet {
                println!("🤖 [BOT] Đạt mức trần chip, đổi thành ALL-IN {} chip", max_total_bet);
                type_bet_amount(&mut enigo, max_total_bet);
            } else {
                println!("🤖 [BOT] Thao tác: RAISE lên {} chip", final_bet);
                type_bet_amount(&mut enigo, final_bet);
            }
        }
    }
}

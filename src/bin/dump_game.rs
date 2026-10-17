use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::{json, Value};

fn get_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

// =========================================================================
// 1. APACHE THRIFT BINARY PROTOCOL DECODER (DÀNH CHO POKERSTARS)
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

    pub fn get_i64(&self) -> Option<i64> {
        match self {
            ThriftVal::I64(v) => Some(*v),
            ThriftVal::I32(v) => Some(*v as i64),
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

pub fn format_pokerstars_card(suit: i32, rank: i32) -> String {
    let r_str = match rank {
        0 => "2", 1 => "3", 2 => "4", 3 => "5", 4 => "6", 5 => "7",
        6 => "8", 7 => "9", 8 => "T", 9 => "J", 10 => "Q", 11 => "K", 12 => "A",
        _ => "?",
    };
    let s_str = match suit {
        0 => "♣", // Clubs
        1 => "♦", // Diamonds
        2 => "♥", // Hearts
        3 => "♠", // Spades
        _ => "?",
    };
    format!("{}{}", r_str, s_str)
}

fn inspect_pokerstars_packet(idx: usize, val: &Value) {
    let dir = val.get("direction").and_then(|v| v.as_str()).unwrap_or("UNKNOWN");
    let format = val.get("format").and_then(|v| v.as_str()).unwrap_or("unknown");
    let _ws_url = val.get("url").and_then(|v| v.as_str()).unwrap_or("");
    let dir_icon = if dir == "INCOMING" { "⬇️ [SERVER -> CLIENT]" } else { "⬆️ [CLIENT -> SERVER]" };

    match format {
        "binary_hex" => {
            let hex_str = match val.get("data").and_then(|v| v.as_str()) {
                Some(h) if h.len() >= 4 => h,
                _ => return,
            };

            let bytes: Vec<u8> = (0..hex_str.len())
                .step_by(2)
                .filter_map(|i| {
                    if i + 2 <= hex_str.len() {
                        u8::from_str_radix(&hex_str[i..i + 2], 16).ok()
                    } else {
                        None
                    }
                })
                .collect();

            if bytes.len() < 2 { return; }
            let opcode = u16::from_be_bytes([bytes[0], bytes[1]]);

            // Giải mã Thrift payload (loại bỏ 2 byte opcode)
            if let Some((thrift_data, _)) = parse_thrift_val(12, &bytes[2..], 0) {
                let table_id = thrift_data.get_field(1).and_then(|v| v.get_str()).unwrap_or("");

                match opcode {
                    0x006a => { // THÔNG TIN BÀN CHƠI (TABLE INFO)
                        if let Some(t_info) = thrift_data.get_field(2) {
                            let name = t_info.get_field(1).and_then(|v| v.get_str()).unwrap_or("?");
                            let seats = t_info.get_field(4).and_then(|v| v.get_i32()).unwrap_or(0);
                            let sb = t_info.get_field(7).and_then(|v| v.get_field(1)).and_then(|v| v.get_i32()).unwrap_or(0);
                            let bb = t_info.get_field(7).and_then(|v| v.get_field(2)).and_then(|v| v.get_i32()).unwrap_or(0);
                            println!("\n🏛️ [GÓI #{}] [POKERSTARS BÀN CHƠI]: \"{}\" | Bàn {}-Max | Blinds: {} / {} chip", idx, name, seats, sb, bb);
                        }
                    }
                    0x006c => { // CẬP NHẬT GHẾ & NGƯỜI CHƠI (PLAYER SEAT)
                        if let Some(p_data) = thrift_data.get_field(2) {
                            let seat = p_data.get_field(1).and_then(|v| v.get_i32()).unwrap_or(-1);
                            if let Some(info) = p_data.get_field(3) {
                                let pname = info.get_field(1).and_then(|v| v.get_str()).unwrap_or("?");
                                let country = info.get_field(2).and_then(|v| v.get_str()).unwrap_or("??");
                                let stack = info.get_field(7).and_then(|v| v.get_i32()).unwrap_or(0);
                                let cur_bet = p_data.get_field(4).and_then(|v| v.get_i32()).unwrap_or(0);
                                println!("👤 [GÓI #{}] [NGƯỜI CHƠI] Ghế #{}: {:<14} ({}) | Stack: {:>10} chip | Cược: {:>7} chip", idx, seat, pname, country, stack, cur_bet);
                            }
                        }
                    }
                    0x0078 => { // TỔNG TIỀN GÀ (POT & RAKE)
                        if let Some(pots) = thrift_data.get_field(2).and_then(|v| v.get_list()) {
                            let total_pot: i32 = pots.iter()
                                .filter_map(|p| p.get_field(2).and_then(|v| v.get_i32()))
                                .sum();
                            println!("🏆 [GÓI #{}] [TỔNG POT]: {} chip", idx, total_pot);
                        }
                    }
                    0x0079 => { // MẢNG CƯỢC TRÊN BÀN (TABLE BETS)
                        if let Some(bets) = thrift_data.get_field(3).and_then(|v| v.get_list()) {
                            let bet_vals: Vec<i32> = bets.iter().filter_map(|v| v.get_i32()).collect();
                            println!("💰 [GÓI #{}] [CƯỢC TRÊN BÀN THEO GHẾ]: {:?}", idx, bet_vals);
                        }
                    }
                    0x0071 => { // BÀI CHUNG (COMMUNITY CARDS)
                        if let Some(cards) = thrift_data.get_field(2).and_then(|v| v.get_list()) {
                            let mut card_strs = Vec::new();
                            for c in cards {
                                let suit = c.get_field(2).and_then(|v| v.get_i32()).unwrap_or(-1);
                                let rank = c.get_field(3).and_then(|v| v.get_i32()).unwrap_or(-1);
                                card_strs.push(format_pokerstars_card(suit, rank));
                            }
                            println!("🎴 [GÓI #{}] [BÀI CHUNG FLOP/TURN/RIVER]: [ {} ]", idx, card_strs.join("  "));
                        }
                    }
                    0x006f => { // LƯỢT ĐÁNH & ĐỒNG HỒ (ACTION TURN)
                        let seat = thrift_data.get_field(2).and_then(|v| v.get_i32()).unwrap_or(-1);
                        let timer = thrift_data.get_field(3).and_then(|v| v.get_i32()).unwrap_or(0);
                        println!("👉 [GÓI #{}] [LƯỢT HÀNH ĐỘNG]: Ghế #{} (Thời gian: {:.1}s)", idx, seat, timer as f64 / 1000.0);
                    }
                    0x00b2 => { // THAM GIA BÀN (JOIN TABLE)
                        let ring = thrift_data.get_field(4).and_then(|v| v.get_str()).unwrap_or("");
                        let curr = thrift_data.get_field(5).and_then(|v| v.get_str()).unwrap_or("");
                        let balance = thrift_data.get_field(6).and_then(|v| v.get_field(1)).and_then(|v| v.get_i32()).unwrap_or(0);
                        println!("🚪 [GÓI #{}] [VÀO BÀN]: Bàn {} | Loại: {} ({}) | Số dư: {} chip", idx, table_id, ring, curr, balance);
                    }
                    _ => {
                        println!("📦 [GÓI #{}] {} Opcode: 0x{:04x} (Size: {} bytes) | Bàn: {}", idx, dir_icon, opcode, bytes.len(), table_id);
                    }
                }
            } else {
                println!("📦 [GÓI #{}] {} Opcode: 0x{:04x} (Size: {} bytes)", idx, dir_icon, opcode, bytes.len());
            }
        }
        "json" => {
            println!("📄 [GÓI #{}] {} JSON Payload", idx, dir_icon);
        }
        "text" => {
            if let Some(text) = val.get("data").and_then(|v| v.as_str()) {
                if text.contains("MESSAGE") && text.contains("userId") {
                    println!("👤 [GÓI #{}] [THÔNG TIN TÀI KHOẢN]: Tìm thấy thông tin user trong STOMP message", idx);
                }
            }
        }
        _ => {}
    }
}

fn inspect_pokernow_summary(idx: usize, val: &Value, current_hand_id: &mut String) {
    let arr = match val.as_array() {
        Some(a) if a.len() >= 2 => a,
        _ => return,
    };

    let event_name = arr[0].as_str().unwrap_or("unknown");
    let payload = match event_name {
        "registered" => arr[1].get("gameState").unwrap_or(&arr[1]),
        _ => &arr[1],
    };

    if let Some(hi) = payload.get("hI").and_then(|v| v.as_str()) {
        if *current_hand_id != hi {
            *current_hand_id = hi.to_string();
            println!("\n🎲 ------------------------------------------------------------");
            println!("🎲 [POKERNOW VÁN MỚI] Hand ID: {} (Gói #{})", hi, idx);
            println!("🎲 ------------------------------------------------------------");
        }
    }

    if let Some(gt) = payload.get("gT").and_then(|v| v.as_array()) {
        if gt.len() >= 2 {
            let st_str = match gt[1].as_u64() {
                Some(0) => "Preflop",
                Some(1) => "Flop",
                Some(2) => "Turn",
                Some(3) => "River",
                Some(4) => "Showdown",
                _ => "Unknown",
            };
            println!("🎴 [STREET UPDATE]: {}", st_str);
        }
    }

    if let Some(p_obj) = payload.get("players").and_then(|v| v.as_object()) {
        println!("👥 [GÓI #{} CÓ BLOCK 'players'] Cập nhật {} người chơi:", idx, p_obj.len());
        for (pid, pval) in p_obj {
            let name = pval.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            let stack = pval.get("stack").and_then(|v| v.as_u64()).unwrap_or(0);
            let seat = pval.get("seat").or_else(|| pval.get("seatNum")).and_then(|v| v.as_u64()).unwrap_or(99);
            println!("   • Ghế #{} {:<15} (ID: {}) -> stack = {}", seat, name, pid, stack);
        }
    }

    if let Some(tb_obj) = payload.get("tB").and_then(|v| v.as_object()) {
        print!("💰 [GÓI #{} 'tB' CƯỢC TRÊN BÀN]: ", idx);
        for (pid, bval) in tb_obj {
            let amt = bval.as_u64().unwrap_or(0);
            print!("| {}: {} chip ", pid, amt);
        }
        println!("|");
    }
}

// Chế độ phân tích file dump offline
fn parse_offline_dump_file(path: &str) {
    println!("🔍 Đang phân tích file dump: {}", path);
    let file = File::open(path).expect("❌ Không thể mở file dump!");
    let reader = BufReader::new(file);

    let mut count = 0;
    for line in reader.lines().flatten() {
        if let Ok(item) = serde_json::from_str::<Value>(&line) {
            count += 1;
            let idx = item.get("packet_idx").and_then(|v| v.as_u64()).unwrap_or(count) as usize;
            let payload = item.get("payload").unwrap_or(&item);
            inspect_pokerstars_packet(idx, payload);
        }
    }
    println!("\n✅ Đã trích xuất xong toàn bộ {} gói tin từ '{}'!", count, path);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Kiểm tra chế độ offline parse: cargo run --bin dump_game -- --parse [file.jsonl]
    if args.len() >= 3 && args[1] == "--parse" {
        parse_offline_dump_file(&args[2]);
        return;
    } else if args.len() == 2 && args[1].ends_with(".jsonl") {
        // Nếu file đã tồn tại và user muốn parse:
        if std::path::Path::new(&args[1]).exists() {
            println!("💡 Phát hiện file '{}' đã có sẵn. Bạn có muốn phân tích file này không?", args[1]);
            println!("👉 Chạy: cargo run --bin dump_game -- --parse {}", args[1]);
        }
    }

    let dump_path = args.get(1).cloned().unwrap_or_else(|| "pokerstars_dump.jsonl".to_string());

    println!("============================================================");
    println!("📡 [POKERSTARS & POKERNOW UNIVERSAL PACKET DUMPER & PARSER] 🕵️‍♂️");
    println!("============================================================");
    println!("📁 Ghi dữ liệu raw vào: {}", dump_path);
    println!("🌐 Đang lắng nghe tại: http://127.0.0.1:3000/packet");
    println!("💡 Hướng dẫn:");
    println!("   1. Để phân tích file đã dump: cargo run --bin dump_game -- --parse {}", dump_path);
    println!("   2. Để bắt trực tiếp: Chạy lệnh này và mở PokerStars Web.");
    println!("============================================================\n");

    let listener = TcpListener::bind("127.0.0.1:3000")
        .expect("❌ Không thể bind cổng 3000! Hãy tắt receiver/tournament nếu đang chạy.");

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&dump_path)
        .expect("❌ Không thể tạo file log!");

    let mut packet_count = 0usize;
    let mut current_hand_id = String::new();

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

                    if !trimmed.is_empty() {
                        packet_count += 1;
                        let ts = get_timestamp_ms();

                        if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                            let record = json!({
                                "packet_idx": packet_count,
                                "timestamp_ms": ts,
                                "payload": val
                            });
                            if let Ok(line) = serde_json::to_string(&record) {
                                let _ = writeln!(log_file, "{}", line);
                                let _ = log_file.flush();
                            }

                            if val.get("source").and_then(|v| v.as_str()) == Some("pokerstars_web") {
                                inspect_pokerstars_packet(packet_count, &val);
                            } else {
                                inspect_pokernow_summary(packet_count, &val, &mut current_hand_id);
                            }
                        }
                    }

                    let resp_body = r#"{"status":"ok"}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                        resp_body.len(),
                        resp_body
                    );
                    let _ = stream.write_all(response.as_bytes());
                    break;
                }
            }
        }
    }
}

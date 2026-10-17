use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::{json, Value};

fn get_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dump_path = args.get(1).cloned().unwrap_or_else(|| "game_dump.jsonl".to_string());

    println!("============================================================");
    println!("📡 [POKERNOW PACKET DUMPER & PROTOCOL INSPECTOR] 🕵️‍♂️");
    println!("============================================================");
    println!("📁 Ghi dữ liệu raw vào: {}", dump_path);
    println!("🌐 Đang lắng nghe tại: http://127.0.0.1:3000/packet");
    println!("💡 Hướng dẫn:");
    println!("   1. Chạy tool này (nó sẽ thay thế receiver2.rs lắng nghe ở cổng 3000).");
    println!("   2. Vào bàn chơi trên trình duyệt, chơi thử 1-2 ván (có đối thủ cược hoặc All-in).");
    println!("   3. Quan sát log tóm tắt trực tiếp trên màn hình terminal.");
    println!("   4. Xong ván, bấm Ctrl + C để dừng. File '{}' sẽ chứa 100% gói tin raw!", dump_path);
    println!("============================================================\n");

    let listener = TcpListener::bind("127.0.0.1:3000").expect("❌ Không thể bind cổng 3000! Hãy tắt receiver2.rs trước nếu đang chạy.");

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&dump_path)
        .expect("❌ Không thể tạo file log!");

    let mut packet_count = 0usize;
    let mut current_hand_id = String::new();

    for mut stream in listener.incoming().flatten() {
        let mut raw_data = Vec::with_capacity(16384);
        let mut temp_buf = [0u8; 8192];

        while let Ok(n) = stream.read(&mut temp_buf) {
            if n == 0 { break; }
            raw_data.extend_from_slice(&temp_buf[..n]);

            if let Some(header_end) = raw_data.windows(4).position(|w| w == b"\r\n\r\n") {
                let header_str = String::from_utf8_lossy(&raw_data[..header_end]);

                // Xử lý CORS OPTIONS
                if header_str.starts_with("OPTIONS") {
                    let response = "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
                    let _ = stream.write_all(response.as_bytes());
                    break;
                }

                // Đọc Content-Length
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
                            // Ghi nguyên bản raw packet vào file JSONL
                            let record = json!({
                                "packet_idx": packet_count,
                                "timestamp_ms": ts,
                                "raw": val
                            });
                            if let Ok(line) = serde_json::to_string(&record) {
                                let _ = writeln!(log_file, "{}", line);
                                let _ = log_file.flush();
                            }

                            // In tóm tắt phân tích ra màn hình để người dùng theo dõi
                            inspect_and_print_summary(packet_count, &val, &mut current_hand_id);
                        } else {
                            println!("⚠️ [GÓI #{}] Không parse được JSON: {}", packet_count, trimmed);
                        }
                    }

                    // Phản hồi OK về cho Tampermonkey
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

fn inspect_and_print_summary(idx: usize, val: &Value, current_hand_id: &mut String) {
    let arr = match val.as_array() {
        Some(a) if a.len() >= 2 => a,
        _ => return,
    };

    let event_name = arr[0].as_str().unwrap_or("unknown");
    let payload = match event_name {
        "registered" => arr[1].get("gameState").unwrap_or(&arr[1]),
        _ => &arr[1],
    };

    // Kiểm tra Ván mới
    if let Some(hi) = payload.get("hI").and_then(|v| v.as_str()) {
        if *current_hand_id != hi {
            *current_hand_id = hi.to_string();
            println!("\n🎲 ------------------------------------------------------------");
            println!("🎲 [VÁN MỚI] Hand ID: {} (Gói #{})", hi, idx);
            println!("🎲 ------------------------------------------------------------");
        }
    }

    // Street
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

    // Block "players"
    if let Some(p_obj) = payload.get("players").and_then(|v| v.as_object()) {
        println!("👥 [GÓI #{} CÓ BLOCK 'players'] Cập nhật {} người chơi:", idx, p_obj.len());
        for (pid, pval) in p_obj {
            let name = pval.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            let stack = pval.get("stack").and_then(|v| v.as_u64()).unwrap_or(0);
            let seat = pval.get("seat").or_else(|| pval.get("seatNum")).and_then(|v| v.as_u64()).unwrap_or(99);
            println!("   • Ghế #{} {:<15} (ID: {}) -> stack trong block players = {}", seat, name, pid, stack);
        }
    }

    // Block "tB" (table bets)
    if let Some(tb_obj) = payload.get("tB").and_then(|v| v.as_object()) {
        print!("💰 [GÓI #{} 'tB' CƯỢC TRÊN BÀN]: ", idx);
        for (pid, bval) in tb_obj {
            let amt = bval.as_u64().unwrap_or(0);
            print!("| {}: {} chip ", pid, amt);
        }
        println!("|");
    }

    // Block "pGS" (player status: allIn, fold, ...)
    if let Some(pgs_obj) = payload.get("pGS").and_then(|v| v.as_object()) {
        print!("⚡ [GÓI #{} 'pGS' TRẠNG THÁI]: ", idx);
        for (pid, sval) in pgs_obj {
            print!("| {}: {} ", pid, sval.as_str().unwrap_or("?"));
        }
        println!("|");
    }

    // Block "mAVTB" (Hero max available bet)
    if let Some(mavtb) = payload.get("mAVTB").and_then(|v| v.as_u64()) {
        println!("🎯 [GÓI #{} 'mAVTB' HERO MAX BET]: {} chip", idx, mavtb);
    }

    // Lượt người chơi
    if let Some(actor) = payload.get("pITT").or_else(|| payload.get("cPI")).and_then(|v| v.as_str()) {
        println!("👉 [LƯỢT HÀNH ĐỘNG]: Người chơi ID [{}]", actor);
    }
}

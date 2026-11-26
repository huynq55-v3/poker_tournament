# Poker Tournament & Lounge Deep CFR (2–10 Max)

Hệ thống Deep CFR & Subgame Solver cho Texas Hold'em 2–10 người chơi với **3 binary chuẩn hóa**:
- `train`: Huấn luyện Deep CFR theo chế độ **Tournament (ICM)** hoặc **Chip-EV / Lounge Cash (12–300+ BB)**
- `play`: Live Assistant + Hybrid Subgame Solver kết nối Tampermonkey (`http://127.0.0.1:3000`) cho cả **Tournament** và **Lounge Cash**
- `poker_gui`: Giao diện eframe/egui chơi thử offline & huấn luyện nền

---

## 1. Huấn luyện (`train`)

### Chế độ Tournament (ICM + Multi-Depth SNG)
```bash
# Train mới từ đầu (8-Max mặc định, ICM, lưu deep_cfr_model.json / deep_cfr_strategy.json)
cargo run --release --bin train -- --mode tournament --players 10 --sparring --fresh

# Train tiếp (resume) từ deep_cfr_last_model.json hoặc checkpoint bất kỳ
cargo run --release --bin train -- --mode tournament --players 10 --iters 2000

# Chạy thêm giải đấu mô phỏng cuối đợt huấn luyện
cargo run --release --bin train -- --mode tournament --players 8 --iters 500 --final-sim
```

### Chế độ Lounge Cash Game (10-Max, Chip-EV, 12–300+ BB)
```bash
# Train mới từ đầu (linh hoạt 2..10 người + Sparring, lưu deep_cfr_lounge_model.json)
cargo run --release --bin train -- --mode chip-ev --players 10 --sparring --fresh

# Khóa cứng 100% ván là 10 người (--fixed-players)
cargo run --release --bin train -- --mode chip-ev --players 10 --fixed-players --sparring --fresh

# Resume Lounge từ deep_cfr_lounge_last_model.json / best model
cargo run --release --bin train -- --mode chip-ev --players 10 --iters 2000
```

> Dùng `cargo run --release --bin train -- --help` để xem toàn bộ siêu tham số (`--warmup`, `--traversals`, `--train-steps`, `--hidden`, `--lr`, `--bench-every`, `--snapshot-every`, `--out`, ...).

---

## 2. Chơi Live trên PokerNow (`play`)

*(Yêu cầu đã có bộ model đúng mode: `deep_cfr_*.json` cho Tournament hoặc `deep_cfr_lounge_*.json` cho Lounge/Chip-EV. Binary không tự fallback chéo mode để tránh đánh nhầm chiến thuật.)*

### Lounge Cash (Chip-EV, mặc định Subgame Solver `depth=4`, `iters=120`, `min_pot=15 BB`)
```bash
# Auto-Play chế độ Silent DOM qua Tampermonkey (cổng 3000)
cargo run --release --bin play -- --mode chip-ev --game-url "https://www.pokernow.com/games/<ROOM_ID>"

# Chỉ phân tích & gợi ý EV/GTO trên terminal, KHÔNG tự bấm nút
cargo run --release --bin play -- --mode chip-ev --advice-only --game-url "https://www.pokernow.com/games/<ROOM_ID>"

# Tùy chỉnh Subgame Solver hoặc tắt Solver (chạy mạng thuần)
cargo run --release --bin play -- --mode chip-ev --search --depth 4 --iters 160 --game-url "https://www.pokernow.com/games/<ROOM_ID>"
cargo run --release --bin play -- --mode chip-ev --no-solver --game-url "https://www.pokernow.com/games/<ROOM_ID>"

# Chế độ trộn bài an toàn (--guarded) hoặc Pure GTO Roll (--roll)
cargo run --release --bin play -- --mode chip-ev --guarded --game-url "https://www.pokernow.com/games/<ROOM_ID>"
cargo run --release --bin play -- --mode chip-ev --roll --game-url "https://www.pokernow.com/games/<ROOM_ID>"
```

### Tournament ICM (mặc định Subgame Solver `depth=3`, `iters=60`, `min_pot=12 BB`)
```bash
# Auto-Play Tournament ICM qua Tampermonkey (cổng 3000)
cargo run --release --bin play -- --mode tournament --game-url "https://www.pokernow.com/games/<ROOM_ID>"

# Advice-only (không tự click) với Guarded Mixing
cargo run --release --bin play -- --mode tournament --guarded --advice-only --game-url "https://www.pokernow.com/games/<ROOM_ID>"
```

---

## 3. Chơi / Test Offline trên GUI (`poker_gui`)

```bash
# Chế độ Lounge Cash 10-Max
cargo run --release --bin poker_gui -- --lounge --seats 10

# Chế độ Tournament 10-Max
cargo run --release --bin poker_gui -- --seats 10
```

---

## 4. Tampermonkey Bridge Scripts

Các script Tampermonkey trong thư mục gốc dùng để kết nối PokerNow với `play` qua `http://127.0.0.1:3000`:
- `tamplermonkey_script1`, `tamplermonkey_script2`, `tamplermonkey_script3`

---

## 5. Kiểm tra & Chạy Test

```bash
cargo check --all-targets
cargo test
```

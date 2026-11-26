# Poker Tournament & Lounge Deep CFR (2–10 Max)

## Tournament Solver
```bash
cargo run --release --bin tournament_solver -- --depth 6 --iters 250
```

## Train Lounge Cash Game (10-Max, Chip-EV, 12–300+ BB)
```bash
# Train mới từ đầu (mạng [256, 256, 256], linh hoạt 2..10 người + Sparring)
cargo run --release --bin train_lounge -- --players 10 --sparring --fresh

# Khóa cứng 100% ván là 10 người
cargo run --release --bin train_lounge -- --players 10 --fixed-players --sparring --fresh

# Train tiếp (resume) từ deep_cfr_lounge_last_model.json / best model
cargo run --release --bin train_lounge -- --players 10 --iters 2000
```

## Play Lounge trên PokerNow (`play_lounge`)
*(Yêu cầu đã train xong `deep_cfr_lounge_model.json`. Mặc định bật Hybrid Chip-EV Subgame Solver: `depth=4`, `iters=120`, `min-pot=15 BB`)*
```bash
# Auto-Play chế độ Silent DOM (gửi lệnh qua Tampermonkey, cổng 3000)
cargo run --release --bin play_lounge -- --game-url "https://www.pokernow.com/games/<ROOM_ID>"

# Chế độ chỉ xem phân tích EV/GTO, không tự động bấm nút
cargo run --release --bin play_lounge -- --advice-only --game-url "https://www.pokernow.com/games/<ROOM_ID>"

# Tùy chỉnh Subgame Solver (ép search mọi vòng, chỉnh depth/iters, hoặc tắt solver chạy mạng thuần)
cargo run --release --bin play_lounge -- --search --depth 4 --iters 160 --game-url "https://www.pokernow.com/games/<ROOM_ID>"
cargo run --release --bin play_lounge -- --no-solver --game-url "https://www.pokernow.com/games/<ROOM_ID>"

# Chế độ trộn bài an toàn (Guarded Mixing) hoặc Pure GTO Roll
cargo run --release --bin play_lounge -- --guarded --game-url "https://www.pokernow.com/games/<ROOM_ID>"
cargo run --release --bin play_lounge -- --roll --game-url "https://www.pokernow.com/games/<ROOM_ID>"
```

## Chơi / Test offline trên GUI (`poker_gui`)
```bash
# Chế độ Lounge Cash 10-Max
cargo run --release --bin poker_gui -- --lounge --seats 10

# Chế độ Tournament 10-Max
cargo run --release --bin poker_gui -- --seats 10
```

## Train Tournament (10-Max ICM)
```bash
cargo run --release --bin poker_tournament -- --players 10 --sparring --fresh
```

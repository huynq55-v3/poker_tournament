# Poker Tournament AI - Deep CFR Engine 🚀

High-Performance Texas Hold'em 2-8 Player GTO Self-Play Simulator with Tournament ICM & Multi-Depth Heads-Up Booster.

## Huấn luyện mới từ đầu (Fresh Training)
```bash
cargo run --release
```

## Huấn luyện tiếp tục (Resume / Continue Training)

Binary chuyên dụng [`src/bin/resume.rs`](file:///home/huy/Documents/poker_tournament/src/bin/resume.rs) hỗ trợ tự động nhận diện checkpoint mới nhất và số iteration bắt đầu:

### 1. Train tiếp tục tự động từ mô hình hiện tại (deep_cfr_model.json):
```bash
cargo run --release --bin resume -- --iters 500
```

### 2. Resume từ một checkpoint cụ thể trong `checkpoints/`:
```bash
cargo run --release --bin resume -- --from checkpoints/deep_cfr_iter01300_model.json --iters 300
```

### 3. Train thử nghiệm độc lập ra bộ file riêng (không ghi đè Champion chính):
```bash
cargo run --release --bin resume -- --from deep_cfr_model.json --out test_run --iters 200
```

### Xem tất cả tùy chọn:
```bash
cargo run --release --bin resume -- --help
```

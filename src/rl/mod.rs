//! Bot RL heads-up (transformer + PPO tự chơi), thử nghiệm so với DeepCFR.
//!
//! - `encoding`: mã hóa trạng thái thành chuỗi token (luôn được biên dịch, không cần `burn`).
//! - Mô hình / huấn luyện: cần feature `rl`.

pub mod encoding;

#[cfg(feature = "rl")]
pub mod model;

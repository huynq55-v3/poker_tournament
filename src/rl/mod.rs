//! Bot RL heads-up (transformer + PPO tự chơi), thử nghiệm so với DeepCFR.
//!
//! - `encoding`: mã hóa trạng thái thành chuỗi token (luôn được biên dịch, không cần `burn`).
//! - Mô hình / huấn luyện: cần feature `rl`.

pub mod encoding;

#[cfg(feature = "rl")]
pub mod model;
#[cfg(feature = "rl")]
pub mod selfplay;

/// Thiết bị chạy mô hình, chọn bằng biến môi trường `RL_DEVICE`:
/// `gpu` (mặc định: CUDA khi build với `rl-cuda`, ngược lại Vulkan) hoặc `cpu`.
#[cfg(feature = "rl")]
pub fn device_from_env() -> burn::tensor::Device {
    use burn::tensor::Device;
    match std::env::var("RL_DEVICE").unwrap_or_default().to_lowercase().as_str() {
        "cpu" => Device::flex(),
        _ => {
            #[cfg(feature = "rl-cuda")]
            {
                return Device::cuda(0);
            }
            #[allow(unreachable_code)]
            Device::vulkan(burn::tensor::DeviceKind::DefaultDevice)
        }
    }
}

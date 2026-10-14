//! Thuật toán Malmuth-Harville tính ICM cho Sit & Go Tournament (Zero-Sum Ranking 8-Max & Short-Handed)

/// Cơ cấu phân bổ điểm/thưởng cho bàn N người trong giải đấu Zero-Sum Ranking:
/// Thang đo luôn đối xứng quanh 0.0 (Zero-Sum: sum = 0.0), biên độ đồng nhất [-0.48, +0.48]
/// giúp gradient của mạng nơ-ron ổn định tuyệt đối trên mọi quy mô bàn chơi.
pub const SNG_PAYOUTS_2MAX: [f32; 2] = [0.48, -0.48];
pub const SNG_PAYOUTS_3MAX: [f32; 3] = [0.48, 0.0, -0.48];
pub const SNG_PAYOUTS_4MAX: [f32; 4] = [0.48, 0.16, -0.16, -0.48];
pub const SNG_PAYOUTS_5MAX: [f32; 5] = [0.48, 0.20, 0.0, -0.20, -0.48];
pub const SNG_PAYOUTS_6MAX: [f32; 6] = [0.48, 0.24, 0.12, -0.12, -0.24, -0.48];
pub const SNG_PAYOUTS_7MAX: [f32; 7] = [0.48, 0.24, 0.14, 0.0, -0.14, -0.24, -0.48];
pub const SNG_PAYOUTS_8MAX: [f32; 8] = [0.48, 0.24, 0.16, 0.12, -0.12, -0.16, -0.24, -0.48];

/// Lấy mảng payout Zero-Sum chuẩn hóa cho bàn `n` người (n <= 8).
/// Zero allocation: Trả về slice tĩnh không cấp phát heap.
#[inline]
pub fn get_sng_payouts(n: usize) -> &'static [f32] {
    match n {
        0 | 1 => &[],
        2 => &SNG_PAYOUTS_2MAX,
        3 => &SNG_PAYOUTS_3MAX,
        4 => &SNG_PAYOUTS_4MAX,
        5 => &SNG_PAYOUTS_5MAX,
        6 => &SNG_PAYOUTS_6MAX,
        7 => &SNG_PAYOUTS_7MAX,
        _ => &SNG_PAYOUTS_8MAX,
    }
}

/// Tính giá trị kỳ vọng giải thưởng (ICM EV) cho từng người chơi
/// dựa trên phân bổ stack hiện tại của bàn bằng quy hoạch động Malmuth-Harville (DP over subsets).
/// Nếu có người chơi bị loại (stack <= 0), họ sẽ nhận các thứ hạng cuối cùng của giải đấu.
///
/// Tự động đảm bảo tính chất Zero-Sum: Nếu `payouts.len() != stacks.len()`,
/// hàm tự động sử dụng bảng điểm chuẩn hóa `get_sng_payouts(n)` để triệt tiêu lệch pha kỳ vọng.
pub fn calculate_icm_ev(stacks: &[f32], payouts: &[f32]) -> Vec<f32> {
    let n = stacks.len();
    if n == 0 {
        return Vec::new();
    }
    let total_chips: f32 = stacks.iter().sum();
    if total_chips <= 0.0 {
        return vec![0.0; n];
    }

    // Tự động chuẩn hóa payout tương ứng với quy mô bàn n người chơi
    let effective_payouts: &[f32] = if payouts.len() == n {
        payouts
    } else {
        get_sng_payouts(n)
    };

    let mut alive_indices = Vec::with_capacity(n);
    let mut dead_indices = Vec::with_capacity(n);

    for (i, &st) in stacks.iter().enumerate() {
        if st > 0.0 {
            alive_indices.push(i);
        } else {
            dead_indices.push(i);
        }
    }

    let mut ev = vec![0.0f32; n];

    // Trường hợp 1: Tất cả người chơi đều còn chip (> 0)
    if dead_indices.is_empty() {
        return calculate_icm_all_alive(stacks, effective_payouts);
    }

    let m = alive_indices.len();
    if m == 0 {
        return ev;
    }

    // Trường hợp 2: Có người chơi bị loại (stack <= 0)
    // Người bị loại nhận các thứ hạng cuối cùng [m .. n-1] (ví dụ: Hạng 8, Hạng 7...)
    let dead_payout_sum: f32 = (m..n)
        .map(|r| if r < effective_payouts.len() { effective_payouts[r] } else { 0.0 })
        .sum();
    let dead_ev = dead_payout_sum / dead_indices.len() as f32;
    for &idx in &dead_indices {
        ev[idx] = dead_ev;
    }

    // Những người còn sống (> 0 chip) chia nhau các thứ hạng đầu tiên [0 .. m-1]
    let alive_payouts = &effective_payouts[..m.min(effective_payouts.len())];
    let alive_stacks: Vec<f32> = alive_indices.iter().map(|&i| stacks[i]).collect();
    let alive_evs = calculate_icm_all_alive(&alive_stacks, alive_payouts);

    for (k, &orig_idx) in alive_indices.iter().enumerate() {
        ev[orig_idx] = alive_evs[k];
    }

    ev
}

/// Thuật toán Malmuth-Harville DP tính ICM khi toàn bộ người chơi trong `stacks` đều còn chip (> 0.0).
fn calculate_icm_all_alive(stacks: &[f32], payouts: &[f32]) -> Vec<f32> {
    let n = stacks.len();
    let num_places = payouts.len().min(n);
    if num_places == 0 {
        return vec![0.0; n];
    }

    let max_mask = 1 << n;
    let mut ev = vec![0.0f32; n];

    // 🚀 ZERO-ALLOCATION FAST PATH: Với giải đấu SNG 8-max (n <= 8),
    // max_mask tối đa 256, toàn bộ mảng DP và sum_remaining được đặt trên Stack (1 KB bộ nhớ),
    // triệt tiêu hoàn toàn chi phí malloc/free heap trong các luồng Rayon!
    if n <= 8 {
        let mut dp = [0.0f32; 256];
        dp[0] = 1.0;

        let mut sum_remaining = [0.0f32; 256];
        for (mask, slot) in sum_remaining[..max_mask].iter_mut().enumerate() {
            let mut s = 0.0f32;
            for (i, &st) in stacks.iter().enumerate() {
                if (mask & (1 << i)) == 0 {
                    s += st;
                }
            }
            *slot = s;
        }

        for mask in 0..max_mask {
            let prob = dp[mask];
            if prob <= 1e-12 {
                continue;
            }

            let rank = mask.count_ones() as usize;
            if rank >= num_places {
                continue;
            }

            let rem_chips = sum_remaining[mask];
            if rem_chips <= 0.0 {
                continue;
            }

            let payout = payouts[rank];

            for (i, &st) in stacks.iter().enumerate() {
                if (mask & (1 << i)) == 0 {
                    let cond_prob = st / rem_chips;
                    let step_prob = prob * cond_prob;
                    ev[i] += step_prob * payout;

                    let next_mask = mask | (1 << i);
                    dp[next_mask] += step_prob;
                }
            }
        }
        return ev;
    }

    // Fallback heap allocation cho trường hợp mở rộng > 8 người
    let mut dp = vec![0.0f32; max_mask];
    dp[0] = 1.0;

    let mut sum_remaining = vec![0.0f32; max_mask];
    for (mask, slot) in sum_remaining.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for (i, &st) in stacks.iter().enumerate() {
            if (mask & (1 << i)) == 0 {
                s += st;
            }
        }
        *slot = s;
    }

    for mask in 0..max_mask {
        let prob = dp[mask];
        if prob <= 1e-12 {
            continue;
        }

        let rank = mask.count_ones() as usize;
        if rank >= num_places {
            continue;
        }

        let rem_chips = sum_remaining[mask];
        if rem_chips <= 0.0 {
            continue;
        }

        let payout = payouts[rank];

        for (i, &st) in stacks.iter().enumerate() {
            if (mask & (1 << i)) == 0 {
                let cond_prob = st / rem_chips;
                let step_prob = prob * cond_prob;
                ev[i] += step_prob * payout;

                let next_mask = mask | (1 << i);
                dp[next_mask] += step_prob;
            }
        }
    }

    ev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icm_symmetry_zero_sum() {
        let stacks = vec![1000.0; 8];
        let evs = calculate_icm_ev(&stacks, &SNG_PAYOUTS_8MAX);
        assert_eq!(evs.len(), 8);
        for &e in &evs {
            assert!(e.abs() < 1e-4, "Equal stacks must have ~0 EV in zero sum");
        }
    }

    #[test]
    fn test_icm_chipleader_vs_shortstack() {
        let stacks = vec![5000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0, 300.0, 200.0];
        let evs = calculate_icm_ev(&stacks, &SNG_PAYOUTS_8MAX);
        // Chip leader phải có EV dương lớn
        assert!(evs[0] > 0.20, "Leader EV = {}", evs[0]);
        // Người ít chip nhất phải có EV âm lớn (gần -0.48)
        assert!(evs[7] < -0.20, "Shortstack EV = {}", evs[7]);
        // Tổng EV phải bảo toàn (bằng tổng payouts)
        let total_ev: f32 = evs.iter().sum();
        let total_payout: f32 = SNG_PAYOUTS_8MAX.iter().sum();
        assert!((total_ev - total_payout).abs() < 1e-3, "EV sum {} vs payout sum {}", total_ev, total_payout);
    }

    #[test]
    fn test_icm_eliminated_player_gets_last_place_payout() {
        // 7 người còn chip, 1 người All-In thua (0 chip)
        let stacks = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 2000.0, 0.0];
        let evs = calculate_icm_ev(&stacks, &SNG_PAYOUTS_8MAX);
        assert_eq!(evs.len(), 8);
        // Người bị loại (0 chip) phải nhận mức phạt Hạng 8 (-0.48 / -9.03 điểm)
        assert!((evs[7] - (-0.48)).abs() < 1e-4, "Eliminated player EV must be -0.48, got {}", evs[7]);
        // Tổng EV toàn bàn vẫn phải Zero-Sum tuyệt đối
        let total_ev: f32 = evs.iter().sum();
        assert!(total_ev.abs() < 1e-4, "Total EV must be 0.0, got {}", total_ev);
    }

    #[test]
    fn test_icm_multiple_eliminated_players_share_bottom_payouts() {
        // 6 người còn chip, 2 người bị loại (0 chip)
        let stacks = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 3000.0, 0.0, 0.0];
        let evs = calculate_icm_ev(&stacks, &SNG_PAYOUTS_8MAX);
        assert_eq!(evs.len(), 8);
        // 2 người bị loại chia nhau Hạng 7 (-0.24) và Hạng 8 (-0.48) => Trung bình: -0.36
        assert!((evs[6] - (-0.36)).abs() < 1e-4, "Eliminated player 6 EV = {}", evs[6]);
        assert!((evs[7] - (-0.36)).abs() < 1e-4, "Eliminated player 7 EV = {}", evs[7]);
        // Tổng EV toàn bàn bảo toàn Zero-Sum
        let total_ev: f32 = evs.iter().sum();
        assert!(total_ev.abs() < 1e-4, "Total EV must be 0.0, got {}", total_ev);
    }

    #[test]
    fn test_icm_all_table_sizes_zero_sum() {
        for n in 2..=8 {
            let payouts = get_sng_payouts(n);
            assert_eq!(payouts.len(), n);
            let payout_sum: f32 = payouts.iter().sum();
            assert!(payout_sum.abs() < 1e-4, "Payout sum for N={} must be 0, got {}", n, payout_sum);

            // 1. Equal stacks
            let stacks = vec![1000.0; n];
            let evs = calculate_icm_ev(&stacks, payouts);
            let total_ev: f32 = evs.iter().sum();
            assert!(total_ev.abs() < 1e-4, "Equal stacks N={} total EV must be 0, got {}", n, total_ev);

            // 2. Unequal stacks
            let unequal_stacks: Vec<f32> = (1..=n).map(|i| (i * 500) as f32).collect();
            let evs_unequal = calculate_icm_ev(&unequal_stacks, payouts);
            let total_ev_unequal: f32 = evs_unequal.iter().sum();
            assert!(total_ev_unequal.abs() < 1e-4, "Unequal stacks N={} total EV must be 0, got {}", n, total_ev_unequal);
        }
    }

    #[test]
    fn test_icm_short_handed_elimination_and_adaptation() {
        // Bàn 4 người nhưng caller truyền SNG_PAYOUTS_8MAX:
        // calculate_icm_ev phải tự động dùng SNG_PAYOUTS_4MAX ([0.48, 0.16, -0.16, -0.48])
        let stacks_4 = vec![1500.0, 1000.0, 500.0, 0.0];
        let evs = calculate_icm_ev(&stacks_4, &SNG_PAYOUTS_8MAX);
        assert_eq!(evs.len(), 4);
        
        // Người bị loại (0 chip) phải bị phạt -0.48 thay vì dương điểm
        assert!((evs[3] - (-0.48)).abs() < 1e-4, "Eliminated player at N=4 must get -0.48, got {}", evs[3]);
        
        // Tổng EV toàn bàn 4 người phải bảo toàn Zero-Sum
        let total_ev: f32 = evs.iter().sum();
        assert!(total_ev.abs() < 1e-4, "Total EV for N=4 must be 0.0, got {}", total_ev);

        // Bàn 6 người: 1 người 0 chip
        let stacks_6 = vec![2000.0, 1500.0, 1000.0, 800.0, 700.0, 0.0];
        let evs_6 = calculate_icm_ev(&stacks_6, &SNG_PAYOUTS_8MAX);
        assert_eq!(evs_6.len(), 6);
        assert!((evs_6[5] - (-0.48)).abs() < 1e-4, "Eliminated player at N=6 must get -0.48, got {}", evs_6[5]);
        let total_ev_6: f32 = evs_6.iter().sum();
        assert!(total_ev_6.abs() < 1e-4, "Total EV for N=6 must be 0.0, got {}", total_ev_6);
    }
}


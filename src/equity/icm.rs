//! Thuật toán Malmuth-Harville tính ICM cho Sit & Go Tournament (Zero-Sum Ranking 8-Max)

/// Cơ cấu phân bổ điểm/thưởng cho 8 người trong giải đấu Zero-Sum Ranking (Dãy điều hòa 12 : 6 : 4 : 3):
/// Hạng 1: +48% (+9.03), Hạng 2: +24% (+4.52), Hạng 3: +16% (+3.01), Hạng 4: +12% (+2.26)
/// Hạng 5: -12% (-2.26), Hạng 6: -16% (-3.01), Hạng 7: -24% (-4.52), Hạng 8: -48% (-9.03)
pub const SNG_PAYOUTS_8MAX: [f32; 8] = [0.48, 0.24, 0.16, 0.12, -0.12, -0.16, -0.24, -0.48];

/// Tính giá trị kỳ vọng giải thưởng (ICM EV) cho từng người chơi
/// dựa trên phân bổ stack hiện tại của bàn bằng quy hoạch động Malmuth-Harville (DP over subsets).
/// Nếu có người chơi bị loại (stack <= 0), họ sẽ nhận các thứ hạng cuối cùng của giải đấu.
pub fn calculate_icm_ev(stacks: &[f32], payouts: &[f32]) -> Vec<f32> {
    let n = stacks.len();
    if n == 0 {
        return Vec::new();
    }
    let total_chips: f32 = stacks.iter().sum();
    if total_chips <= 0.0 {
        return vec![0.0; n];
    }

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
        return calculate_icm_all_alive(stacks, payouts);
    }

    let m = alive_indices.len();
    if m == 0 {
        return ev;
    }

    // Trường hợp 2: Có người chơi bị loại (stack <= 0)
    // Người bị loại nhận các thứ hạng cuối cùng [m .. n-1] (ví dụ: Hạng 8, Hạng 7...)
    let dead_payout_sum: f32 = (m..n)
        .map(|r| if r < payouts.len() { payouts[r] } else { 0.0 })
        .sum();
    let dead_ev = dead_payout_sum / dead_indices.len() as f32;
    for &idx in &dead_indices {
        ev[idx] = dead_ev;
    }

    // Những người còn sống (> 0 chip) chia nhau các thứ hạng đầu tiên [0 .. m-1]
    let alive_payouts = &payouts[..m.min(payouts.len())];
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

    // Dynamic Programming qua tập con (bitmask)
    // dp[mask] = xác suất tập hợp người chơi trong `mask` đã chiếm các vị trí 1 .. count_ones(mask)
    let max_mask = 1 << n;
    let mut dp = vec![0.0f32; max_mask];
    dp[0] = 1.0;

    let mut ev = vec![0.0f32; n];

    // Tính trước tổng số chip còn lại của những người chưa có thứ hạng (chưa nằm trong mask)
    let mut sum_remaining = vec![0.0f32; max_mask];
    for mask in 0..max_mask {
        let mut s = 0.0f32;
        for i in 0..n {
            if (mask & (1 << i)) == 0 {
                s += stacks[i];
            }
        }
        sum_remaining[mask] = s;
    }

    for mask in 0..max_mask {
        let prob = dp[mask];
        if prob <= 1e-12 {
            continue;
        }

        let rank = mask.count_ones() as usize; // Vị trí tiếp theo cần xác định (0 là Hạng 1)
        if rank >= num_places {
            continue;
        }

        let rem_chips = sum_remaining[mask];
        if rem_chips <= 0.0 {
            continue;
        }

        let payout = payouts[rank];

        for i in 0..n {
            if (mask & (1 << i)) == 0 {
                let cond_prob = stacks[i] / rem_chips;
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
}

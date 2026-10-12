//! Thuật toán Malmuth-Harville tính ICM cho Sit & Go Tournament

/// Cơ cấu phân bổ điểm/thưởng cho Top 4 trong giải 8 người
/// (Top 1: 40%, Top 2: 30%, Top 3: 20%, Top 4: 10%, còn lại 0)
pub const SNG_PAYOUTS_8MAX: [f32; 4] = [0.50, 0.25, 0.15, 0.10];

/// Tính giá trị phần trăm giải thưởng (ICM EV từ 0.0 đến 1.0) cho từng người chơi
/// dựa trên phân bổ stack hiện tại của bàn.
pub fn calculate_icm_ev(stacks: &[f32], payouts: &[f32]) -> Vec<f32> {
    let n = stacks.len();
    let num_payouts = payouts.len().min(n);
    let total_chips: f32 = stacks.iter().sum();

    if total_chips <= 0.0 {
        return vec![0.0; n];
    }

    let mut ev = vec![0.0f32; n];

    // Xác suất về Nhất (Vị trí 1) tỉ lệ thuận với số chip
    let mut prob_pos1 = vec![0.0f32; n];
    for i in 0..n {
        prob_pos1[i] = stacks[i] / total_chips;
        ev[i] += prob_pos1[i] * payouts[0];
    }

    if num_payouts == 1 {
        return ev;
    }

    // Đệ quy/xác suất có điều kiện cho Vị trí 2, 3, 4 (Malmuth-Harville)
    for i in 0..n {
        if stacks[i] <= 0.0 { continue; }

        for j in 0..n {
            if i == j || stacks[j] <= 0.0 { continue; }
            let rem2 = total_chips - stacks[i];
            let p_j_2 = (stacks[j] / rem2) * prob_pos1[i];
            ev[j] += p_j_2 * payouts[1];

            if num_payouts > 2 {
                for k in 0..n {
                    if k == i || k == j || stacks[k] <= 0.0 { continue; }
                    let rem3 = rem2 - stacks[j];
                    let p_k_3 = (stacks[k] / rem3) * p_j_2;
                    ev[k] += p_k_3 * payouts[2];

                    if num_payouts > 3 {
                        for m in 0..n {
                            if m == i || m == j || m == k || stacks[m] <= 0.0 { continue; }
                            let rem4 = rem3 - stacks[k];
                            let p_m_4 = (stacks[m] / rem4) * p_k_3;
                            ev[m] += p_m_4 * payouts[3];
                        }
                    }
                }
            }
        }
    }

    ev
}

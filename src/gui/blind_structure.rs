use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct BlindLevel {
    pub level_num: usize,
    pub sb: u32,
    pub bb: u32,
}

#[derive(Debug, Clone)]
pub struct TournamentBlindSchedule {
    pub levels: Vec<BlindLevel>,
    pub current_idx: usize,
    pub hands_per_level: usize,
    pub hands_in_current_level: usize,
    pub total_hands_played: usize,
    pub level_duration: Duration,
    pub current_level_started_at: Instant,
    pub is_paused: bool,
    pub paused_at: Option<Instant>,
}

impl TournamentBlindSchedule {
    pub fn new(num_seats: usize) -> Self {
        // Thang Blind chuẩn mực theo giải đấu Tournament Sit & Go
        let raw_levels = vec![
            (50, 100),
            (75, 150),
            (100, 200),
            (150, 300),
            (200, 400),
            (300, 600),
            (500, 1000),
            (800, 1600),
            (1200, 2400),
            (2000, 4000),
            (3000, 6000),
            (5000, 10000),
        ];

        let levels = raw_levels
            .into_iter()
            .enumerate()
            .map(|(i, (sb, bb))| BlindLevel {
                level_num: i + 1,
                sb,
                bb,
            })
            .collect();

        let hands_per_level = num_seats.clamp(2, 8); // Mặc định: 1 vòng cược quanh bàn (8 ván cho bàn 8 người)

        Self {
            levels,
            current_idx: 0,
            hands_per_level,
            hands_in_current_level: 0,
            total_hands_played: 0,
            level_duration: Duration::from_secs(300),
            current_level_started_at: Instant::now(),
            is_paused: false,
            paused_at: None,
        }
    }

    /// Ghi nhận hoàn thành 1 ván bài. Tự động tăng Level Blind khi đủ số ván cố định!
    pub fn on_hand_completed(&mut self) -> bool {
        self.total_hands_played += 1;
        self.hands_in_current_level += 1;

        if self.hands_in_current_level >= self.hands_per_level && self.current_idx + 1 < self.levels.len() {
            self.current_idx += 1;
            self.hands_in_current_level = 0;
            self.current_level_started_at = Instant::now();
            true
        } else {
            false
        }
    }

    /// Số ván còn lại trước khi tăng Level Blind tiếp theo
    pub fn hands_remaining_in_level(&self) -> usize {
        self.hands_per_level.saturating_sub(self.hands_in_current_level)
    }

    /// Chuỗi thông tin tiến độ Blind theo ván bài
    pub fn formatted_hand_progress(&self) -> String {
        format!(
            "Ván {}/{} (còn {} ván tăng Blind)",
            self.hands_in_current_level, self.hands_per_level, self.hands_remaining_in_level()
        )
    }

    /// Thời gian đã trôi qua của level hiện tại (đóng băng khi đang pause)
    fn elapsed(&self) -> Duration {
        match self.paused_at {
            Some(t) => t.saturating_duration_since(self.current_level_started_at),
            None => self.current_level_started_at.elapsed(),
        }
    }

    pub fn pause(&mut self) {
        if !self.is_paused {
            self.is_paused = true;
            self.paused_at = Some(Instant::now());
        }
    }

    /// ✅ FIX: dịch mốc bắt đầu level về sau đúng bằng thời gian đã pause
    pub fn resume(&mut self) {
        if self.is_paused {
            if let Some(t) = self.paused_at.take() {
                self.current_level_started_at += t.elapsed();
            }
            self.is_paused = false;
        }
    }

    pub fn toggle_pause(&mut self) {
        if self.is_paused {
            self.resume();
        } else {
            self.pause();
        }
    }

    /// Chuyển sang level kế tiếp ngay lập tức. Trả về true nếu thành công.
    pub fn advance_level_manually(&mut self) -> bool {
        if self.current_idx + 1 < self.levels.len() {
            self.current_idx += 1;
            self.current_level_started_at = Instant::now();
            if self.is_paused {
                self.paused_at = Some(Instant::now());
            }
            true
        } else {
            false
        }
    }

    /// Update schedule; advances level if the duration elapsed
    pub fn update(&mut self) -> bool {
        if self.is_paused {
            return false;
        }

        if self.elapsed() >= self.level_duration && self.current_idx + 1 < self.levels.len() {
            self.current_idx += 1;
            self.current_level_started_at = Instant::now();
            return true;
        }
        false
    }

    pub fn current_level(&self) -> BlindLevel {
        self.levels[self.current_idx]
    }

    pub fn time_remaining(&self) -> Duration {
        let elapsed = self.elapsed();
        if elapsed >= self.level_duration {
            Duration::ZERO
        } else {
            self.level_duration - elapsed
        }
    }

    pub fn formatted_time_remaining(&self) -> String {
        let rem = self.time_remaining().as_secs();
        format!("{:02}:{:02}", rem / 60, rem % 60)
    }

    pub fn next_level(&self) -> Option<BlindLevel> {
        if self.current_idx + 1 < self.levels.len() {
            Some(self.levels[self.current_idx + 1])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blind_schedule_progression() {
        let mut schedule8 = TournamentBlindSchedule::new(8);
        assert_eq!(schedule8.levels.len(), 12);
        assert_eq!(schedule8.current_level().sb, 50);
        assert_eq!(schedule8.current_level().bb, 100);
        assert_eq!(schedule8.hands_per_level, 8);

        // Chơi 7 ván -> chưa tăng level
        for _ in 0..7 {
            assert!(!schedule8.on_hand_completed());
        }
        assert_eq!(schedule8.current_level().level_num, 1);
        assert_eq!(schedule8.hands_remaining_in_level(), 1);

        // Ván thứ 8 -> Tăng Level 2 (75/150)!
        assert!(schedule8.on_hand_completed());
        assert_eq!(schedule8.current_level().level_num, 2);
        assert_eq!(schedule8.current_level().sb, 75);
        assert_eq!(schedule8.current_level().bb, 150);
        assert_eq!(schedule8.hands_remaining_in_level(), 8);
    }

    #[test]
    fn test_pause_freezes_timer() {
        let mut s = TournamentBlindSchedule::new(6);
        s.current_level_started_at = Instant::now() - Duration::from_secs(100);
        s.pause();
        std::thread::sleep(Duration::from_millis(30));
        assert!(!s.update());
        s.resume();
        assert!(s.time_remaining() > Duration::from_secs(190));
        assert!(!s.update());
    }
}

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
    pub level_duration: Duration,
    pub current_level_started_at: Instant,
    pub is_paused: bool,
    pub paused_at: Option<Instant>,
}

impl TournamentBlindSchedule {
    pub fn new(num_seats: usize) -> Self {
        let mut raw_levels = vec![
            (10, 20),
            (20, 40),
            (30, 60),
            (50, 100),
            (75, 150),
            (100, 200),
            (150, 300),
            (250, 500),
        ];

        if num_seats > 6 {
            raw_levels.push((300, 600));
        }

        let levels = raw_levels
            .into_iter()
            .enumerate()
            .map(|(i, (sb, bb))| BlindLevel {
                level_num: i + 1,
                sb,
                bb,
            })
            .collect();

        Self {
            levels,
            current_idx: 0,
            level_duration: Duration::from_secs(300),
            current_level_started_at: Instant::now(),
            is_paused: false,
            paused_at: None,
        }
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
        let mut schedule6 = TournamentBlindSchedule::new(6);
        assert_eq!(schedule6.levels.len(), 8);
        assert_eq!(schedule6.current_level().sb, 10);
        assert_eq!(schedule6.current_level().bb, 20);

        let schedule8 = TournamentBlindSchedule::new(8);
        assert_eq!(schedule8.levels.len(), 9);
        assert_eq!(schedule8.levels.last().unwrap().bb, 600);
        assert_eq!(schedule8.levels.last().unwrap().sb, 300);

        schedule6.current_level_started_at = Instant::now() - Duration::from_secs(305);
        let leveled_up = schedule6.update();
        assert!(leveled_up);
        assert_eq!(schedule6.current_level().sb, 20);
        assert_eq!(schedule6.current_level().bb, 40);
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

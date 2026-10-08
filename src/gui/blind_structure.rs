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
    pub elapsed_paused: Duration,
}

impl TournamentBlindSchedule {
    pub fn new(num_seats: usize) -> Self {
        // Blind structure: 10/20, 20/40, 30/60, 50/100, 75/150, 100/200, 150/300, 250/500, (300/600 in > 6 seats table)
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
            level_duration: Duration::from_secs(300), // 5 minutes
            current_level_started_at: Instant::now(),
            is_paused: false,
            elapsed_paused: Duration::ZERO,
        }
    }

    /// Update schedule; advances level if 5 minutes elapsed
    pub fn update(&mut self) -> bool {
        if self.is_paused {
            return false;
        }

        let elapsed = self.current_level_started_at.elapsed();
        if elapsed >= self.level_duration && self.current_idx + 1 < self.levels.len() {
            self.current_idx += 1;
            self.current_level_started_at = Instant::now();
            return true; // Leveled up!
        }
        false
    }

    pub fn current_level(&self) -> BlindLevel {
        self.levels[self.current_idx]
    }

    pub fn time_remaining(&self) -> Duration {
        let elapsed = self.current_level_started_at.elapsed();
        if elapsed >= self.level_duration {
            Duration::ZERO
        } else {
            self.level_duration - elapsed
        }
    }

    pub fn formatted_time_remaining(&self) -> String {
        let rem = self.time_remaining().as_secs();
        let minutes = rem / 60;
        let seconds = rem % 60;
        format!("{:02}:{:02}", minutes, seconds)
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
        // Table 6 seats
        let mut schedule6 = TournamentBlindSchedule::new(6);
        assert_eq!(schedule6.levels.len(), 8);
        assert_eq!(schedule6.current_level().sb, 10);
        assert_eq!(schedule6.current_level().bb, 20);

        // Table 8 seats (>6 seats has 300/600 level)
        let schedule8 = TournamentBlindSchedule::new(8);
        assert_eq!(schedule8.levels.len(), 9);
        assert_eq!(schedule8.levels.last().unwrap().bb, 600);
        assert_eq!(schedule8.levels.last().unwrap().sb, 300);

        // Advance level test
        schedule6.current_level_started_at = Instant::now() - Duration::from_secs(305);
        let leveled_up = schedule6.update();
        assert!(leveled_up);
        assert_eq!(schedule6.current_level().sb, 20);
        assert_eq!(schedule6.current_level().bb, 40);
    }
}


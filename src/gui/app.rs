use crate::deep_cfr::DeepCFRPolicy;
use crate::environment::action::{Action, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::Position;
use crate::gui::blind_structure::TournamentBlindSchedule;
use crate::neural_network::MLP;
use crate::poker_core::card::{Card, Suit};
use crate::self_play::Policy;
use eframe::egui::{self, Color32, RichText, Stroke, Vec2};
use rand::thread_rng;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

pub struct PokerGuiApp {
    pub env: Environment,
    pub blind_schedule: TournamentBlindSchedule,
    pub policy: DeepCFRPolicy,
    pub num_seats: usize,
    pub hero_seat: usize,
    pub button_idx: usize,
    pub last_bot_act_time: Instant,
    pub bot_delay_ms: u64,
    pub auto_play_bots: bool,
    pub hand_number: usize,
    pub hand_in_progress: bool,
    pub winner_announcement: Option<String>,
    pub action_logs: Vec<String>,
}

impl PokerGuiApp {
    pub fn new(num_seats: usize) -> Self {
        let num_seats = num_seats.clamp(2, 8);
        let schedule = TournamentBlindSchedule::new(num_seats);
        let current_blind = schedule.current_level();

        let initial_stack = 200 * current_blind.bb; // 200 BB starting stack
        let stacks = vec![initial_stack; num_seats];
        let button_idx = 0;

        let mut env = Environment::new(num_seats, &stacks, current_blind.bb, current_blind.sb, button_idx);
        let mut rng = thread_rng();
        env.reset_hand(&mut rng, button_idx);

        // Mark seat 0 as Hero (human)
        env.state.players[0].is_bot = false;

        // Create Deep CFR Policy (can load checkpoint or init)
        let mlp = MLP::new(&[126, 256, 256, 128, 6]);
        let policy = DeepCFRPolicy::new(Arc::new(RwLock::new(mlp)), false);

        let mut app = Self {
            env,
            blind_schedule: schedule,
            policy,
            num_seats,
            hero_seat: 0,
            button_idx,
            last_bot_act_time: Instant::now(),
            bot_delay_ms: 600,
            auto_play_bots: true,
            hand_number: 1,
            hand_in_progress: true,
            winner_announcement: None,
            action_logs: vec!["🎲 Tournament started! Blinds: 10/20.".to_string()],
        };

        app.log_hand_start();
        app
    }

    fn log_hand_start(&mut self) {
        let bl = self.blind_schedule.current_level();
        self.action_logs.push(format!(
            "--- Hand #{} started (Level {}: {}/{}) ---",
            self.hand_number, bl.level_num, bl.sb, bl.bb
        ));
    }

    fn start_next_hand(&mut self) {
        self.hand_number += 1;
        self.button_idx = (self.button_idx + 1) % self.num_seats;

        // Check for eliminated players, rebuy if hero broke
        if self.env.state.players[self.hero_seat].stack == 0 {
            self.env.state.players[self.hero_seat].stack = 100 * self.blind_schedule.current_level().bb;
            self.action_logs.push("💰 Hero re-bought for 100 BB!".to_string());
        }

        // Top-up bankrupt bots so table remains active
        for (i, p) in self.env.state.players.iter_mut().enumerate() {
            if i != self.hero_seat && p.stack < self.blind_schedule.current_level().bb {
                p.stack = 100 * self.blind_schedule.current_level().bb;
            }
        }

        let curr_bl = self.blind_schedule.current_level();
        self.env.state.bb_size = curr_bl.bb;
        self.env.state.sb_size = curr_bl.sb;

        let mut rng = thread_rng();
        self.env.reset_hand(&mut rng, self.button_idx);
        self.env.state.players[self.hero_seat].is_bot = false;

        self.hand_in_progress = true;
        self.winner_announcement = None;
        self.log_hand_start();
    }

    fn execute_action(&mut self, action: Action) {
        let actor = self.env.state.current_player_idx;
        let is_hero = actor == self.hero_seat;
        let name = if is_hero { "Hero".to_string() } else { format!("Bot_{}", actor) };

        let result = self.env.step(action);
        self.action_logs.push(format!(
            "[{:?}] {} performed {}",
            self.env.state.street, name, action.name()
        ));

        match result {
            Ok(is_hand_over) => {
                if is_hand_over {
                    self.hand_in_progress = false;
                    self.check_showdown_result();
                }
            }
            Err(e) => {
                self.action_logs.push(format!("⚠️ Error: {}", e));
            }
        }
    }

    fn check_showdown_result(&mut self) {
        let mut winners = Vec::new();
        let mut max_stack = 0u32;
        for (_, p) in self.env.state.players.iter().enumerate() {
            if p.is_in_hand() && p.stack > max_stack {
                max_stack = p.stack;
            }
        }
        for (i, p) in self.env.state.players.iter().enumerate() {
            if p.is_in_hand() {
                winners.push(if i == self.hero_seat { "Hero".to_string() } else { format!("Bot_{}", i) });
            }
        }
        let announcement = format!("🏆 Hand ended! Final active players: {}", winners.join(", "));
        self.action_logs.push(announcement.clone());
        self.winner_announcement = Some(announcement);
    }
}

impl eframe::App for PokerGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1. Update Blind Timer
        if self.blind_schedule.update() {
            let next_bl = self.blind_schedule.current_level();
            self.action_logs.push(format!(
                "🔔 BLINDS INCREASED! Level {}: {}/{}",
                next_bl.level_num, next_bl.sb, next_bl.bb
            ));
            self.env.state.bb_size = next_bl.bb;
            self.env.state.sb_size = next_bl.sb;
        }

        // Request continuous repaint for smooth timer and bot turn delays
        ctx.request_repaint_after(Duration::from_millis(50));

        // 2. Handle Bot Turns automatically if hand is in progress
        if self.hand_in_progress {
            let curr_actor = self.env.state.current_player_idx;
            let is_bot = self.env.state.players[curr_actor].is_bot;

            if is_bot && self.auto_play_bots {
                if self.last_bot_act_time.elapsed() >= Duration::from_millis(self.bot_delay_ms) {
                    let mask = self.env.get_action_mask();
                    let features = self.env.state.encode_features(curr_actor);
                    let bot_action = self.policy.select_action(&features, &mask);
                    self.execute_action(bot_action);
                    self.last_bot_act_time = Instant::now();
                }
            }
        }

        // 3. Render Top Banner (Tournament Timer & Blinds)
        egui::TopBottomPanel::top("top_banner").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading(RichText::new("♠️ TEXAS HOLD'EM DEEP CFR TOURNAMENT").color(Color32::from_rgb(240, 200, 80)).strong());
                ui.separator();

                let curr_bl = self.blind_schedule.current_level();
                ui.label(RichText::new(format!("Level {}", curr_bl.level_num)).strong());
                ui.label(RichText::new(format!("Blinds: {}/{}", curr_bl.sb, curr_bl.bb)).color(Color32::from_rgb(100, 220, 100)).strong());

                ui.separator();
                let timer_text = format!("⏳ Next Blind: {}", self.blind_schedule.formatted_time_remaining());
                ui.label(RichText::new(timer_text).color(Color32::from_rgb(255, 130, 130)).strong());

                if let Some(nxt) = self.blind_schedule.next_level() {
                    ui.label(format!("(Next: {}/{})", nxt.sb, nxt.bb));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⏭️ Next Blind Level").clicked() {
                        if self.blind_schedule.current_idx + 1 < self.blind_schedule.levels.len() {
                            self.blind_schedule.current_idx += 1;
                            self.blind_schedule.current_level_started_at = Instant::now();
                            let bl = self.blind_schedule.current_level();
                            self.env.state.bb_size = bl.bb;
                            self.env.state.sb_size = bl.sb;
                            self.action_logs.push(format!("⏭️ Manually skipped to Level {}: {}/{}", bl.level_num, bl.sb, bl.bb));
                        }
                    }
                    if ui.button(if self.blind_schedule.is_paused { "▶️ Resume Timer" } else { "⏸️ Pause Timer" }).clicked() {
                        self.blind_schedule.is_paused = !self.blind_schedule.is_paused;
                    }
                });
            });
            ui.add_space(4.0);
        });

        // 4. Render Right Sidebar (Logs & GTO Bot Probabilities)
        egui::SidePanel::right("right_panel")
            .min_width(320.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("📊 GTO Decision Probs");
                ui.separator();

                let curr_actor = self.env.state.current_player_idx;
                let mask = self.env.get_action_mask();
                let feat = self.env.state.encode_features(curr_actor);
                let probs = self.policy.get_action_probs(&feat, &mask);

                for a_idx in 0..NUM_ACTIONS {
                    let act = Action::from_index(a_idx).unwrap();
                    let p = probs[a_idx];
                    let is_valid = mask.is_index_valid(a_idx);

                    ui.horizontal(|ui| {
                        let text_color = if is_valid { Color32::WHITE } else { Color32::DARK_GRAY };
                        ui.label(RichText::new(format!("{:12}", act.name())).color(text_color));
                        let bar_color = if is_valid { Color32::from_rgb(60, 160, 240) } else { Color32::GRAY };
                        ui.add(egui::ProgressBar::new(p).text(format!("{:.1}%", p * 100.0)).fill(bar_color));
                    });
                }

                ui.add_space(16.0);
                ui.heading("📜 Hand Logs");
                ui.separator();

                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for log in &self.action_logs {
                            ui.label(RichText::new(log).size(12.0));
                        }
                    });
            });

        // 5. Render Central Table & Action Controls
        egui::CentralPanel::default().show(ctx, |ui| {
            // Main Poker Felt
            let table_rect = ui.available_rect_before_wrap();
            let table_height = table_rect.height() - 140.0;
            let center = egui::pos2(table_rect.center().x, table_rect.min.y + table_height * 0.48);

            // Draw Oval Felt
            let painter = ui.painter();
            let felt_size = Vec2::new(table_rect.width() * 0.88, table_height * 0.82);
            let felt_rect = egui::Rect::from_center_size(center, felt_size);

            painter.rect_filled(felt_rect, 100.0, Color32::from_rgb(24, 76, 42)); // Classic poker green
            painter.rect_stroke(felt_rect, 100.0, Stroke::new(6.0, Color32::from_rgb(180, 140, 60))); // Gold border

            // Draw Center Pot & Community Cards
            let pot_str = format!("POT: {} chips ({:.1} BB)", self.env.state.pot, self.env.state.pot as f32 / self.env.state.bb_size as f32);
            painter.text(
                egui::pos2(center.x, center.y - 45.0),
                egui::Align2::CENTER_CENTER,
                pot_str,
                egui::FontId::proportional(18.0),
                Color32::from_rgb(240, 220, 100),
            );

            // Draw Community Cards
            let card_start_x = center.x - (self.env.state.community_cards.len() as f32 * 28.0);
            for (c_idx, &card) in self.env.state.community_cards.iter().enumerate() {
                let card_pos = egui::pos2(card_start_x + (c_idx as f32 * 56.0), center.y);
                draw_card(painter, card_pos, card, true);
            }

            // Draw Players around Table
            let num_p = self.num_seats;
            let radius_x = felt_size.x * 0.44;
            let radius_y = felt_size.y * 0.42;

            for i in 0..num_p {
                let angle = std::f32::consts::PI * 0.5 + (i as f32 * 2.0 * std::f32::consts::PI / num_p as f32);
                let seat_pos = egui::pos2(center.x + radius_x * angle.cos(), center.y + radius_y * angle.sin());

                let p = &self.env.state.players[i];
                let is_actor = i == self.env.state.current_player_idx && self.hand_in_progress;
                let is_hero = i == self.hero_seat;
                let pos_name = Position::for_seat(i, self.button_idx, num_p);

                // Seat background box
                let seat_rect = egui::Rect::from_center_size(seat_pos, Vec2::new(115.0, 75.0));
                let border_color = if is_actor {
                    Color32::from_rgb(255, 230, 40) // Glowing turn highlight
                } else if is_hero {
                    Color32::from_rgb(80, 180, 255)
                } else {
                    Color32::from_rgb(70, 70, 70)
                };

                let bg_color = if p.is_folded {
                    Color32::from_rgba_unmultiplied(30, 30, 30, 200)
                } else {
                    Color32::from_rgba_unmultiplied(20, 24, 28, 230)
                };

                painter.rect_filled(seat_rect, 8.0, bg_color);
                painter.rect_stroke(seat_rect, 8.0, Stroke::new(if is_actor { 3.0 } else { 1.5 }, border_color));

                // Name & Position
                let p_name = if is_hero { "Hero (You)".to_string() } else { format!("Bot_{}", i) };
                let pos_badge = if i == self.button_idx { format!("{} [D]", format!("{:?}", pos_name)) } else { format!("{:?}", pos_name) };

                painter.text(
                    egui::pos2(seat_pos.x, seat_pos.y - 25.0),
                    egui::Align2::CENTER_CENTER,
                    format!("{} ({})", p_name, pos_badge),
                    egui::FontId::proportional(12.0),
                    Color32::WHITE,
                );

                // Stack
                let stack_bb = p.stack as f32 / self.env.state.bb_size as f32;
                painter.text(
                    egui::pos2(seat_pos.x, seat_pos.y - 8.0),
                    egui::Align2::CENTER_CENTER,
                    format!("{} ({:.1} BB)", p.stack, stack_bb),
                    egui::FontId::proportional(12.0),
                    Color32::from_rgb(120, 230, 120),
                );

                // Hole Cards or Status
                if p.is_folded {
                    painter.text(
                        egui::pos2(seat_pos.x, seat_pos.y + 16.0),
                        egui::Align2::CENTER_CENTER,
                        "FOLDED",
                        egui::FontId::proportional(13.0),
                        Color32::GRAY,
                    );
                } else if p.is_all_in {
                    painter.text(
                        egui::pos2(seat_pos.x, seat_pos.y + 16.0),
                        egui::Align2::CENTER_CENTER,
                        "ALL-IN!",
                        egui::FontId::proportional(13.0),
                        Color32::RED,
                    );
                } else if is_hero || !self.hand_in_progress {
                    // Show Hero's cards or show all cards at showdown
                    if let (Some(c0), Some(c1)) = (p.hole_cards[0], p.hole_cards[1]) {
                        draw_card(painter, egui::pos2(seat_pos.x - 18.0, seat_pos.y + 18.0), c0, false);
                        draw_card(painter, egui::pos2(seat_pos.x + 18.0, seat_pos.y + 18.0), c1, false);
                    }
                } else {
                    // Hidden Bot Cards
                    painter.text(
                        egui::pos2(seat_pos.x, seat_pos.y + 16.0),
                        egui::Align2::CENTER_CENTER,
                        "🂠  🂠",
                        egui::FontId::proportional(18.0),
                        Color32::LIGHT_BLUE,
                    );
                }

                // Bet chips in front of player
                if p.current_bet > 0 {
                    let bet_offset = Vec2::new(-angle.cos() * 45.0, -angle.sin() * 45.0);
                    let bet_pos = seat_pos + bet_offset;
                    painter.text(
                        bet_pos,
                        egui::Align2::CENTER_CENTER,
                        format!("🪙 {}", p.current_bet),
                        egui::FontId::proportional(12.0),
                        Color32::from_rgb(255, 230, 100),
                    );
                }
            }

            // Bottom Player Action Control Dock
            ui.add_space(table_height);
            ui.separator();

            let is_hero_turn = self.env.state.current_player_idx == self.hero_seat && self.hand_in_progress;
            let mask = self.env.get_action_mask();

            ui.horizontal(|ui| {
                if !self.hand_in_progress {
                    if ui.button(RichText::new("🃏 Deal Next Hand").size(18.0).strong().color(Color32::GREEN)).clicked() {
                        self.start_next_hand();
                    }
                    if let Some(ann) = &self.winner_announcement {
                        ui.label(RichText::new(ann).color(Color32::from_rgb(240, 220, 80)).size(15.0).strong());
                    }
                } else if is_hero_turn {
                    ui.label(RichText::new("👉 YOUR TURN:").size(16.0).color(Color32::YELLOW).strong());

                    let to_call = self.env.state.current_highest_bet.saturating_sub(self.env.state.players[self.hero_seat].current_bet);

                    if ui.add_enabled(mask.is_valid(Action::Fold), egui::Button::new("🛑 Fold")).clicked() {
                        self.execute_action(Action::Fold);
                    }

                    let call_label = if to_call == 0 { "🛡️ Check".to_string() } else { format!("🛡️ Call {}", to_call) };
                    if ui.add_enabled(mask.is_valid(Action::CheckCall), egui::Button::new(call_label)).clicked() {
                        self.execute_action(Action::CheckCall);
                    }

                    if ui.add_enabled(mask.is_valid(Action::BetPot33), egui::Button::new("🎯 Bet 33%")).clicked() {
                        self.execute_action(Action::BetPot33);
                    }

                    if ui.add_enabled(mask.is_valid(Action::BetPot66), egui::Button::new("💥 Bet 66%")).clicked() {
                        self.execute_action(Action::BetPot66);
                    }

                    if ui.add_enabled(mask.is_valid(Action::BetPot100), egui::Button::new("💣 Bet 100%")).clicked() {
                        self.execute_action(Action::BetPot100);
                    }

                    if ui.add_enabled(mask.is_valid(Action::AllIn), egui::Button::new(RichText::new("🚀 All-In").color(Color32::RED).strong())).clicked() {
                        self.execute_action(Action::AllIn);
                    }
                } else {
                    let actor = self.env.state.current_player_idx;
                    ui.label(RichText::new(format!("⏳ Bot_{} is thinking...", actor)).size(15.0));

                    if !self.auto_play_bots {
                        if ui.button("▶️ Step Bot").clicked() {
                            let curr_actor = self.env.state.current_player_idx;
                            let feat = self.env.state.encode_features(curr_actor);
                            let bot_action = self.policy.select_action(&feat, &mask);
                            self.execute_action(bot_action);
                        }
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut self.auto_play_bots, "Auto Bot Moves");
                    ui.add(egui::Slider::new(&mut self.bot_delay_ms, 100..=1500).text("Bot Delay (ms)"));
                });
            });
        });
    }
}

fn draw_card(painter: &egui::Painter, pos: egui::Pos2, card: Card, large: bool) {
    let (w, h) = if large { (44.0, 60.0) } else { (30.0, 42.0) };
    let card_rect = egui::Rect::from_center_size(pos, Vec2::new(w, h));

    painter.rect_filled(card_rect, 4.0, Color32::WHITE);
    painter.rect_stroke(card_rect, 4.0, Stroke::new(1.0, Color32::DARK_GRAY));

    let suit_color = match card.suit() {
        Suit::Hearts | Suit::Diamonds => Color32::from_rgb(220, 20, 20),
        Suit::Spades | Suit::Clubs => Color32::from_rgb(20, 20, 20),
    };

    let font_size = if large { 16.0 } else { 12.0 };
    painter.text(
        pos,
        egui::Align2::CENTER_CENTER,
        format!("{}", card),
        egui::FontId::proportional(font_size),
        suit_color,
    );
}

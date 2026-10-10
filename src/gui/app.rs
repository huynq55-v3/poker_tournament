use crate::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use crate::environment::action::{Action, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::{Position, Street, FEATURE_DIM};
use crate::gui::blind_structure::TournamentBlindSchedule;
use crate::neural_network::MLP;
use crate::poker_core::card::{Card, Suit};
use crate::self_play::Policy;
use eframe::egui::{self, Color32, RichText, Stroke, Vec2};
use rand::thread_rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

const ADV_MODEL_PATH: &str = "deep_cfr_model.json";

fn new_untrained_net() -> MLP {
    MLP::new(&[FEATURE_DIM, 256, 256, 128, NUM_ACTIONS])
}

/// Nạp 1 model tùy ý từ đường dẫn chỉ định
fn load_model_from_path(path: &std::path::Path) -> Result<(MLP, String), String> {
    match MLP::load_from_file(path) {
        Ok(adv) if adv.input_dim() == FEATURE_DIM => {
            let filename = path.file_name().unwrap_or_default().to_string_lossy();
            Ok((adv, format!("🟢 Loaded: {}", filename)))
        }
        Ok(adv) => Err(format!(
            "🔴 Input dim không khớp: model có {} chiều, yêu cầu {} chiều!",
            adv.input_dim(),
            FEATURE_DIM
        )),
        Err(e) => Err(format!("🔴 Lỗi đọc file: {}", e)),
    }
}

/// Load advantage net mặc định từ ADV_MODEL_PATH
fn load_models() -> (MLP, Option<MLP>, String) {
    match MLP::load_from_file(ADV_MODEL_PATH) {
        Ok(adv) if adv.input_dim() == FEATURE_DIM => {
            let strat: Option<MLP> = None; // Advantage net (regret matching) chơi sắc bén nhất
            let status = "🟢 Loaded Advantage Net (Regret Matching)";
            (adv, strat, status.to_string())
        }
        Ok(_) => (
            new_untrained_net(),
            None,
            format!("🔴 Model input size != {} (old model). Please retrain.", FEATURE_DIM),
        ),
        Err(_) => (
            new_untrained_net(),
            None,
            "🟡 Untrained Model (Nạp model hoặc train mới)".to_string(),
        ),
    }
}

pub struct PokerGuiApp {
    pub env: Environment,
    pub blind_schedule: TournamentBlindSchedule,
    pub policy: DeepCFRPolicy,
    pub num_seats: usize,
    pub selected_num_seats: usize,
    pub hero_seat: usize,
    pub button_idx: usize,
    pub last_bot_act_time: Instant,
    pub bot_delay_ms: u64,
    pub auto_play_bots: bool,
    pub hand_number: usize,
    pub hand_in_progress: bool,
    pub tournament_over: bool,
    pub winner_announcement: Option<String>,
    pub action_logs: Vec<String>,
    pub model_status: String,
    pub is_training: Arc<AtomicBool>,
    pub train_progress_msg: Arc<RwLock<String>>,
    pub eliminated_logged: Vec<bool>,
}

impl PokerGuiApp {
    pub fn new(num_seats: usize) -> Self {
        let _ = crate::equity::preflop_table(); // Build/load trước để không giật ở frame đầu

        let num_seats = num_seats.clamp(2, 8);
        let schedule = TournamentBlindSchedule::new(num_seats);
        let current_blind = schedule.current_level();

        let initial_stack = 1000u32;
        let stacks = vec![initial_stack; num_seats];
        let button_idx = 0;

        let mut env = Environment::new(num_seats, &stacks, current_blind.bb, current_blind.sb, button_idx);
        let mut rng = thread_rng();
        env.reset_hand(&mut rng, button_idx);

        let (adv, strat, status) = load_models();
        let advantage_net = Arc::new(RwLock::new(adv));
        let policy = DeepCFRPolicy::with_strategy(Arc::clone(&advantage_net), strat, false);

        let mut app = Self {
            env,
            blind_schedule: schedule,
            policy,
            num_seats,
            selected_num_seats: num_seats,
            hero_seat: 0,
            button_idx,
            last_bot_act_time: Instant::now(),
            bot_delay_ms: 600,
            auto_play_bots: true,
            hand_number: 1,
            hand_in_progress: true,
            tournament_over: false,
            winner_announcement: None,
            action_logs: vec![format!(
                "🎲 Started {}-Player Tournament! 1,000 chips each. Blinds: {}/{}.",
                num_seats, current_blind.sb, current_blind.bb
            )],
            model_status: status,
            is_training: Arc::new(AtomicBool::new(false)),
            train_progress_msg: Arc::new(RwLock::new(String::new())),
            eliminated_logged: vec![false; num_seats],
        };

        app.begin_hand();
        app
    }

    /// Mở native file dialog cho phép chọn model .json bất kỳ
    pub fn open_file_and_load_model(&mut self) {
        let file_opt = rfd::FileDialog::new()
            .add_filter("Neural Network Model (*.json)", &["json"])
            .set_title("Chọn model Deep CFR để nạp...")
            .pick_file();

        if let Some(path) = file_opt {
            match load_model_from_path(&path) {
                Ok((adv, status_msg)) => {
                    if let Ok(mut net) = self.policy.advantage_net.write() {
                        *net = adv;
                    }
                    if let Ok(mut strat) = self.policy.strategy_net.write() {
                        *strat = None;
                    }
                    self.model_status = status_msg.clone();
                    self.action_logs.push(format!("💾 Đã nạp thành công: {:?}", path.file_name().unwrap()));
                }
                Err(err_msg) => {
                    self.model_status = err_msg.clone();
                    self.action_logs.push(format!("⚠️ {}", err_msg));
                }
            }
        }
    }

    /// Reset và bắt đầu giải đấu hoàn toàn mới
    pub fn start_new_tournament(&mut self, seats: usize) {
        self.num_seats = seats.clamp(2, 8);
        self.selected_num_seats = self.num_seats;
        self.blind_schedule = TournamentBlindSchedule::new(self.num_seats);
        let current_blind = self.blind_schedule.current_level();

        let stacks = vec![1000u32; self.num_seats];
        self.button_idx = 0;
        self.hand_number = 1;
        self.tournament_over = false;
        self.winner_announcement = None;
        self.eliminated_logged = vec![false; self.num_seats];

        self.env = Environment::new(self.num_seats, &stacks, current_blind.bb, current_blind.sb, self.button_idx);
        let mut rng = thread_rng();
        self.env.reset_hand(&mut rng, self.button_idx);

        self.action_logs.clear();
        self.action_logs.push(format!(
            "🏆 New {}-Player Tournament Started! 1,000 chips each. Blinds: {}/{}.",
            self.num_seats, current_blind.sb, current_blind.bb
        ));
        self.begin_hand();
    }

    fn surviving_players(&self) -> Vec<usize> {
        (0..self.num_seats)
            .filter(|&i| self.env.state.players[i].stack > 0)
            .collect()
    }

    fn log_hand_start(&mut self) {
        let bl = self.blind_schedule.current_level();
        let surv = self.surviving_players().len();
        self.action_logs.push(format!(
            "--- Hand #{} started ({} players left | Level {}: {}/{}) ---",
            self.hand_number, surv, bl.level_num, bl.sb, bl.bb
        ));
    }

    fn begin_hand(&mut self) {
        self.env.state.players[self.hero_seat].is_bot = false;
        self.hand_in_progress = true;
        self.winner_announcement = None;
        self.log_hand_start();

        if self.env.state.street == Street::Showdown {
            self.hand_in_progress = false;
            self.check_showdown_result();
        }
    }

    fn start_next_hand(&mut self) {
        let surviving = self.surviving_players();

        // Chỉ còn 1 người có chip -> Tournament kết thúc
        if surviving.len() <= 1 {
            self.tournament_over = true;
            self.hand_in_progress = false;
            if let Some(&winner) = surviving.first() {
                let win_name = if winner == self.hero_seat {
                    "Hero (You) 🏆".to_string()
                } else {
                    format!("Bot_{}", winner)
                };
                let msg = format!("🏆 TOURNAMENT OVER! {} won all chips and is the Champion!", win_name);
                self.action_logs.push(msg.clone());
                self.winner_announcement = Some(msg);
            }
            return;
        }

        self.hand_number += 1;

        // Tiến button: Chỉ trao button cho người THỰC SỰ CÒN CHIP (stack > 0)
        let curr_btn = self.env.state.button_idx;
        let mut next_btn = (curr_btn + 1) % self.num_seats;
        while self.env.state.players[next_btn].stack == 0 {
            next_btn = (next_btn + 1) % self.num_seats;
        }
        self.button_idx = next_btn;

        let curr_bl = self.blind_schedule.current_level();
        self.env.state.bb_size = curr_bl.bb;
        self.env.state.sb_size = curr_bl.sb;

        let mut rng = thread_rng();
        self.env.reset_hand(&mut rng, self.button_idx);
        self.begin_hand();
    }

    fn player_name(&self, idx: usize) -> String {
        if idx == self.hero_seat {
            "Hero".to_string()
        } else {
            format!("Bot_{}", idx)
        }
    }

    fn execute_action(&mut self, action: Action) {
        let actor = self.env.state.current_player_idx;
        let name = self.player_name(actor);
        let street = self.env.state.street;

        let result = self.env.step(action);
        self.action_logs.push(format!("[{:?}] {} performed {}", street, name, action.name()));

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
        let remaining: Vec<String> = (0..self.num_seats)
            .filter(|&i| self.env.state.players[i].is_in_hand())
            .map(|i| self.player_name(i))
            .collect();
        let announcement = format!("Hand ended. Remaining in hand: {}", remaining.join(", "));
        self.action_logs.push(announcement.clone());

        // CHỈ GHI NHẬN ELIMINATED KHI VÁN BÀI ĐÃ CHIA XONG POT
        // (Ai All-in mà THẮNG thì stack đã > 0, ai All-in mà THUA thì stack mới thực sự = 0)
        for i in 0..self.num_seats {
            let p = &mut self.env.state.players[i];
            if p.stack == 0 && !self.eliminated_logged[i] {
                self.eliminated_logged[i] = true;
                p.is_folded = true;
                p.is_all_in = false;
                p.hole_cards = [None, None];

                let name = if i == self.hero_seat {
                    "Hero (You)".to_string()
                } else {
                    format!("Bot_{}", i)
                };
                self.action_logs.push(format!("💀 {} ran out of chips and is ELIMINATED!", name));
            }
        }

        let surviving = self.surviving_players();
        if surviving.len() <= 1 {
            self.tournament_over = true;
            let winner_idx = surviving.first().copied().unwrap_or(0);
            let win_name = if winner_idx == self.hero_seat {
                "Hero (You) 🏆".to_string()
            } else {
                format!("Bot_{}", winner_idx)
            };
            self.winner_announcement = Some(format!("🏆 TOURNAMENT OVER! {} is the Champion!", win_name));
        } else {
            self.winner_announcement = Some(announcement);
        }
    }

    pub fn reload_model(&mut self) {
        let (adv, strat, status) = load_models();
        if let Ok(mut net) = self.policy.advantage_net.write() {
            *net = adv;
        }
        if let Ok(mut s) = self.policy.strategy_net.write() {
            *s = strat;
        }
        self.model_status = status;
        self.action_logs.push("💾 Neural Network reloaded from disk.".to_string());
    }

    pub fn start_background_training(&mut self) {
        if self.is_training.load(Ordering::SeqCst) {
            return;
        }

        self.is_training.store(true, Ordering::SeqCst);
        let is_training_flag = Arc::clone(&self.is_training);
        let progress_msg = Arc::clone(&self.train_progress_msg);
        let target_adv = Arc::clone(&self.policy.advantage_net);
        let target_strat = Arc::clone(&self.policy.strategy_net);

        std::thread::spawn(move || {
            let mut rng = thread_rng();
            let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &[256, 256, 128], 30_000);
            let total_iters = 10;

            for iter in 1..=total_iters {
                let adv_loss = trainer.step_iteration(20, 40, 128, &mut rng);
                if let Ok(mut msg) = progress_msg.write() {
                    *msg = format!("Iter {}/{}: Adv Loss {:.3}", iter, total_iters, adv_loss);
                }
            }

            if let Ok(trained) = trainer.advantage_net.read() {
                if let Ok(mut live) = target_adv.write() {
                    *live = trained.clone();
                }
            }
            if trainer.strategy_trained {
                if let Ok(trained) = trainer.strategy_net.read() {
                    if let Ok(mut live) = target_strat.write() {
                        *live = Some(trained.clone());
                    }
                }
            }

            if let Ok(mut msg) = progress_msg.write() {
                *msg = "✅ Quick training done. Hot-reloaded (not saved to disk).".to_string();
            }
            is_training_flag.store(false, Ordering::SeqCst);
        });
    }
}

impl eframe::App for PokerGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1. Blind timer
        if self.blind_schedule.update() {
            let nb = self.blind_schedule.current_level();
            self.action_logs.push(format!(
                "🔔 BLINDS UP! Level {}: {}/{} (applies from next hand)",
                nb.level_num, nb.sb, nb.bb
            ));
        }

        ctx.request_repaint_after(Duration::from_millis(50));

        // 2. Bot turns
        if self.hand_in_progress && !self.tournament_over {
            let curr_actor = self.env.state.current_player_idx;
            let is_bot = self.env.state.players[curr_actor].is_bot;
            let is_alive = self.env.state.players[curr_actor].is_active();

            if is_bot && is_alive && self.auto_play_bots
                && self.last_bot_act_time.elapsed() >= Duration::from_millis(self.bot_delay_ms)
            {
                let mask = self.env.get_action_mask();
                let features = self.env.state.encode_features(curr_actor);
                let bot_action = self.policy.select_action(&features, &mask);
                self.execute_action(bot_action);
                self.last_bot_act_time = Instant::now();
            }
        }

        // 3. Top banner
        egui::TopBottomPanel::top("top_banner").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(RichText::new("♠️ TEXAS HOLD'EM TOURNAMENT").color(Color32::from_rgb(240, 200, 80)).strong());
                ui.separator();

                ui.label(RichText::new("👥 Players:").strong());
                if ui.button("➖").clicked() && self.selected_num_seats > 2 {
                    self.selected_num_seats -= 1;
                }
                ui.label(RichText::new(format!("{}", self.selected_num_seats)).size(16.0).color(Color32::from_rgb(100, 200, 255)).strong());
                if ui.button("➕").clicked() && self.selected_num_seats < 8 {
                    self.selected_num_seats += 1;
                }

                if ui.button(RichText::new("🔄 New Tournament").color(Color32::LIGHT_GREEN).strong()).clicked() {
                    self.start_new_tournament(self.selected_num_seats);
                }

                ui.separator();

                let curr_bl = self.blind_schedule.current_level();
                ui.label(RichText::new(format!("Level {}", curr_bl.level_num)).strong());
                ui.label(RichText::new(format!("Blinds: {}/{}", curr_bl.sb, curr_bl.bb)).color(Color32::from_rgb(100, 220, 100)).strong());
                if curr_bl.bb != self.env.state.bb_size {
                    ui.label(RichText::new("(next hand)").size(11.0).color(Color32::from_rgb(255, 200, 80)));
                }

                ui.separator();
                let timer_text = format!("⏳ {}", self.blind_schedule.formatted_time_remaining());
                ui.label(RichText::new(timer_text).color(Color32::from_rgb(255, 130, 130)).strong());

                let surv_count = self.surviving_players().len();
                ui.label(RichText::new(format!("(Alive: {}/{})", surv_count, self.num_seats)).color(Color32::from_rgb(200, 200, 200)));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⏭️ Next Blind").clicked() {
                        if self.blind_schedule.advance_level_manually() {
                            let bl = self.blind_schedule.current_level();
                            self.action_logs.push(format!(
                                "⏭️ Blinds skipped to Level {}: {}/{} (applies from next hand)",
                                bl.level_num, bl.sb, bl.bb
                            ));
                        }
                    }
                    let pause_label = if self.blind_schedule.is_paused { "▶️ Resume" } else { "⏸️ Pause" };
                    if ui.button(pause_label).clicked() {
                        self.blind_schedule.toggle_pause();
                    }
                });
            });
            ui.add_space(4.0);
        });

        // 4. Right sidebar
        egui::SidePanel::right("right_panel")
            .min_width(330.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);

                ui.heading("🧠 Deep CFR Neural Net");
                ui.separator();
                ui.label(RichText::new(&self.model_status).size(12.0));

                ui.horizontal(|ui| {
                    let training_active = self.is_training.load(Ordering::Relaxed);
                    if training_active {
                        ui.add(egui::Spinner::new());
                        if let Ok(msg) = self.train_progress_msg.read() {
                            ui.label(RichText::new(&*msg).color(Color32::from_rgb(255, 200, 60)).size(11.0));
                        }
                    } else {
                        if ui.button("🏋️ Quick Train (10 Iter)").clicked() {
                            self.start_background_training();
                        }
                        if ui.button("📂 Chọn & Nạp Model...").clicked() {
                            self.open_file_and_load_model();
                        }
                        if ui.button("🔄 Reload Mặc Định").clicked() {
                            self.reload_model();
                        }
                    }
                });
                if !self.is_training.load(Ordering::Relaxed) {
                    if let Ok(msg) = self.train_progress_msg.read() {
                        if !msg.is_empty() {
                            ui.label(RichText::new(&*msg).size(11.0).color(Color32::from_rgb(160, 220, 160)));
                        }
                    }
                }

                ui.add_space(12.0);
                ui.heading("📊 Real-time GTO Decision Probs");
                ui.separator();

                let curr_actor = self.env.state.current_player_idx;
                let mask = self.env.get_action_mask();
                let feat = self.env.state.encode_features(curr_actor);
                let probs = self.policy.get_action_probs(&feat, &mask);
                
                // SỬA THÀNH:
                if curr_actor == self.hero_seat && self.hand_in_progress && !self.tournament_over {
                    let raw = self.policy.advantage_net.read().unwrap().forward(&feat);
                    println!("\n🔍 [DEBUG HERO]:");
                    println!("   🃏 Bài tẩy : {:?}", self.env.state.players[curr_actor].hole_cards);
                    println!("   🎭 Action Mask : {:?}", mask.to_bool_array());
                    println!("   🧠 Raw Regrets : {:?}", &raw[..6]);
                    println!("   📊 Probs xuất ra: {:?}", &probs[..6]);
                }

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

                ui.add_space(14.0);
                ui.heading("📜 Tournament Logs");
                ui.separator();

                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for log in &self.action_logs {
                            ui.label(RichText::new(log).size(12.0));
                        }
                    });
            });

        // 5. Central table & controls
        egui::CentralPanel::default().show(ctx, |ui| {
            let table_rect = ui.available_rect_before_wrap();
            let table_height = table_rect.height() - 150.0;
            let center = egui::pos2(table_rect.center().x, table_rect.min.y + table_height * 0.48);

            let painter = ui.painter();
            let felt_size = Vec2::new(table_rect.width() * 0.90, table_height * 0.84);
            let felt_rect = egui::Rect::from_center_size(center, felt_size);

            painter.rect_filled(felt_rect, 120.0, Color32::from_rgb(18, 62, 35));
            painter.rect_stroke(felt_rect, 120.0, Stroke::new(7.0, Color32::from_rgb(190, 150, 70)));

            let pot_str = format!(
                "POT: {} chips ({:.1} BB)",
                self.env.state.pot,
                self.env.state.pot as f32 / self.env.state.bb_size as f32
            );
            painter.text(
                egui::pos2(center.x, center.y - 60.0),
                egui::Align2::CENTER_CENTER,
                pot_str,
                egui::FontId::proportional(22.0),
                Color32::from_rgb(255, 230, 110),
            );

            let num_comm = self.env.state.community_cards.len();
            let card_spacing = 68.0;
            let card_start_x = center.x - ((num_comm as f32 - 1.0).max(0.0) * card_spacing * 0.5);

            for (c_idx, &card) in self.env.state.community_cards.iter().enumerate() {
                let card_pos = egui::pos2(card_start_x + (c_idx as f32 * card_spacing), center.y);
                draw_large_card(painter, card_pos, card);
            }

            let num_p = self.num_seats;
            let radius_x = felt_size.x * 0.44;
            let radius_y = felt_size.y * 0.43;
            let engine_button = self.env.state.button_idx;

            for i in 0..num_p {
                let angle = std::f32::consts::PI * 0.5 + (i as f32 * 2.0 * std::f32::consts::PI / num_p as f32);
                let seat_pos = egui::pos2(center.x + radius_x * angle.cos(), center.y + radius_y * angle.sin());

                let p = &self.env.state.players[i];
                let is_actor = i == self.env.state.current_player_idx && self.hand_in_progress && !self.tournament_over;
                let is_hero = i == self.hero_seat;

                // CHUẨN XÁC: Chỉ Busted khi ván bài đã kết thúc VÀ stack = 0, HOẶC đã fold và stack = 0
                // (Nếu đang all-in trong ván thì chưa tính là busted!)
                let is_busted = p.stack == 0 && (!self.hand_in_progress || p.is_folded);
                let alive_seats: Vec<usize> = self.env.state
    .players
    .iter()
    .enumerate()
    .filter(|(_, p)| p.stack > 0 || p.is_all_in)
    .map(|(idx, _)| idx)
    .collect();
let pos_name = Position::for_seat_alive(i, engine_button, &alive_seats);

                let seat_rect = egui::Rect::from_center_size(seat_pos, Vec2::new(138.0, 96.0));

                if is_busted {
                    painter.rect_filled(seat_rect, 10.0, Color32::from_rgba_unmultiplied(25, 20, 20, 230));
                    painter.rect_stroke(seat_rect, 10.0, Stroke::new(1.5, Color32::from_rgb(100, 30, 30)));

                    let p_name = if is_hero { "Hero (You)".to_string() } else { format!("Bot_{}", i) };
                    painter.text(
                        egui::pos2(seat_pos.x, seat_pos.y - 20.0),
                        egui::Align2::CENTER_CENTER,
                        p_name,
                        egui::FontId::proportional(13.0),
                        Color32::from_rgb(180, 80, 80),
                    );
                    painter.text(
                        egui::pos2(seat_pos.x, seat_pos.y + 6.0),
                        egui::Align2::CENTER_CENTER,
                        "💀 ELIMINATED",
                        egui::FontId::proportional(15.0),
                        Color32::from_rgb(255, 75, 75),
                    );
                } else {
                    let border_color = if is_actor {
                        Color32::from_rgb(255, 215, 0)
                    } else if is_hero {
                        Color32::from_rgb(80, 190, 255)
                    } else {
                        Color32::from_rgb(70, 75, 85)
                    };

                    let bg_color = if p.is_folded {
                        Color32::from_rgba_unmultiplied(18, 18, 20, 220)
                    } else {
                        Color32::from_rgba_unmultiplied(22, 28, 36, 245)
                    };

                    painter.rect_filled(seat_rect, 10.0, bg_color);
                    painter.rect_stroke(seat_rect, 10.0, Stroke::new(if is_actor { 3.0 } else { 1.5 }, border_color));

                    let p_name = if is_hero { "Hero".to_string() } else { format!("Bot_{}", i) };
                    let pos_badge = if i == engine_button {
                        format!("{:?} [BTN]", pos_name)
                    } else {
                        format!("{:?}", pos_name)
                    };

                    painter.text(
                        egui::pos2(seat_pos.x, seat_pos.y - 34.0),
                        egui::Align2::CENTER_CENTER,
                        format!("{} ({})", p_name, pos_badge),
                        egui::FontId::proportional(13.0),
                        if is_hero { Color32::from_rgb(130, 220, 255) } else { Color32::WHITE },
                    );

                    let stack_bb = p.stack as f32 / self.env.state.bb_size as f32;
                    painter.text(
                        egui::pos2(seat_pos.x, seat_pos.y - 17.0),
                        egui::Align2::CENTER_CENTER,
                        format!("{} chips ({:.1} BB)", p.stack, stack_bb),
                        egui::FontId::proportional(12.5),
                        Color32::from_rgb(120, 245, 120),
                    );

                    if p.is_folded {
                        painter.text(
                            egui::pos2(seat_pos.x, seat_pos.y + 16.0),
                            egui::Align2::CENTER_CENTER,
                            "FOLDED",
                            egui::FontId::proportional(15.0),
                            Color32::DARK_GRAY,
                        );
                    } else {
                        let should_reveal_cards =
                            is_hero || !self.hand_in_progress || self.env.state.street == Street::Showdown;

                        if should_reveal_cards {
                            if let (Some(c0), Some(c1)) = (p.hole_cards[0], p.hole_cards[1]) {
                                draw_medium_card(painter, egui::pos2(seat_pos.x - 24.0, seat_pos.y + 16.0), c0);
                                draw_medium_card(painter, egui::pos2(seat_pos.x + 24.0, seat_pos.y + 16.0), c1);
                            }
                        } else {
                            draw_card_back(painter, egui::pos2(seat_pos.x - 20.0, seat_pos.y + 16.0));
                            draw_card_back(painter, egui::pos2(seat_pos.x + 20.0, seat_pos.y + 16.0));
                        }

                        if p.is_all_in {
                            let badge_rect = egui::Rect::from_center_size(
                                egui::pos2(seat_pos.x, seat_pos.y + 36.0),
                                Vec2::new(70.0, 18.0),
                            );
                            painter.rect_filled(badge_rect, 4.0, Color32::from_rgb(220, 30, 30));
                            painter.text(
                                egui::pos2(seat_pos.x, seat_pos.y + 36.0),
                                egui::Align2::CENTER_CENTER,
                                "ALL-IN",
                                egui::FontId::proportional(12.0),
                                Color32::WHITE,
                            );
                        }
                    }

                    if p.current_bet > 0 {
                        let bet_offset = Vec2::new(-angle.cos() * 56.0, -angle.sin() * 56.0);
                        let bet_pos = seat_pos + bet_offset;
                        painter.text(
                            bet_pos,
                            egui::Align2::CENTER_CENTER,
                            format!("🪙 {}", p.current_bet),
                            egui::FontId::proportional(14.0),
                            Color32::from_rgb(255, 235, 100),
                        );
                    }
                }
            }

            // Bottom action dock
            ui.add_space(table_height);
            ui.separator();

            let hero_alive = self.env.state.players[self.hero_seat].stack > 0;
            let is_hero_turn = self.env.state.current_player_idx == self.hero_seat
                && self.hand_in_progress
                && hero_alive
                && !self.tournament_over;
            let mask = self.env.get_action_mask();

            ui.horizontal(|ui| {
                if self.tournament_over {
                    if let Some(ann) = &self.winner_announcement {
                        ui.label(RichText::new(ann).color(Color32::from_rgb(250, 220, 80)).size(17.0).strong());
                    }
                    if ui.button(RichText::new("🔄 Start New Tournament").size(16.0).color(Color32::GREEN).strong()).clicked() {
                        self.start_new_tournament(self.selected_num_seats);
                    }
                } else if !self.hand_in_progress {
                    if ui.button(RichText::new("🃏 Deal Next Hand").size(17.0).strong().color(Color32::GREEN)).clicked() {
                        self.start_next_hand();
                    }
                    if let Some(ann) = &self.winner_announcement {
                        ui.label(RichText::new(ann).color(Color32::from_rgb(250, 220, 80)).size(15.0).strong());
                    }
                    if !hero_alive {
                        ui.label(RichText::new("💀 You have busted! Spectating bots or click 'New Tournament' above.").color(Color32::from_rgb(255, 120, 120)));
                    }
                } else if is_hero_turn {
                    ui.label(RichText::new("👉 YOUR TURN:").size(17.0).color(Color32::YELLOW).strong());

                    let to_call = self
                        .env
                        .state
                        .current_highest_bet
                        .saturating_sub(self.env.state.players[self.hero_seat].current_bet);

                    if ui.add_enabled(mask.is_valid(Action::Fold), egui::Button::new(RichText::new("🛑 Fold").size(15.0))).clicked() {
                        self.execute_action(Action::Fold);
                    }

                    let call_label = if to_call == 0 { "🛡️ Check".to_string() } else { format!("🛡️ Call {}", to_call) };
                    if ui.add_enabled(mask.is_valid(Action::CheckCall), egui::Button::new(RichText::new(call_label).size(15.0))).clicked() {
                        self.execute_action(Action::CheckCall);
                    }

                    if ui.add_enabled(mask.is_valid(Action::BetPot33), egui::Button::new(RichText::new("🎯 Bet 33%").size(15.0))).clicked() {
                        self.execute_action(Action::BetPot33);
                    }

                    if ui.add_enabled(mask.is_valid(Action::BetPot66), egui::Button::new(RichText::new("💥 Bet 66%").size(15.0))).clicked() {
                        self.execute_action(Action::BetPot66);
                    }

                    if ui.add_enabled(mask.is_valid(Action::BetPot100), egui::Button::new(RichText::new("💣 Bet 100%").size(15.0))).clicked() {
                        self.execute_action(Action::BetPot100);
                    }

                    if ui.add_enabled(mask.is_valid(Action::AllIn), egui::Button::new(RichText::new("🚀 All-In").size(15.0).color(Color32::RED).strong())).clicked() {
                        self.execute_action(Action::AllIn);
                    }
                } else {
                    let actor = self.env.state.current_player_idx;
                    ui.label(RichText::new(format!("⏳ Bot_{} is thinking...", actor)).size(16.0));

                    if !self.auto_play_bots && ui.button("▶️ Step Bot").clicked() {
                        let curr_actor = self.env.state.current_player_idx;
                        let feat = self.env.state.encode_features(curr_actor);
                        let bot_action = self.policy.select_action(&feat, &mask);
                        self.execute_action(bot_action);
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

/// Draw large community cards (Width: 58, Height: 84)
fn draw_large_card(painter: &egui::Painter, pos: egui::Pos2, card: Card) {
    let w = 58.0;
    let h = 84.0;
    let card_rect = egui::Rect::from_center_size(pos, Vec2::new(w, h));

    painter.rect_filled(card_rect, 6.0, Color32::WHITE);
    painter.rect_stroke(card_rect, 6.0, Stroke::new(1.5, Color32::from_rgb(160, 160, 160)));

    let suit_color = match card.suit() {
        Suit::Hearts | Suit::Diamonds => Color32::from_rgb(220, 20, 20),
        Suit::Spades | Suit::Clubs => Color32::from_rgb(18, 18, 18),
    };

    painter.text(
        egui::pos2(pos.x - 16.0, pos.y - 24.0),
        egui::Align2::CENTER_CENTER,
        format!("{}", card.rank().char_code()),
        egui::FontId::proportional(20.0),
        suit_color,
    );

    painter.text(
        egui::pos2(pos.x, pos.y + 6.0),
        egui::Align2::CENTER_CENTER,
        card.suit().symbol(),
        egui::FontId::proportional(32.0),
        suit_color,
    );
}

/// Draw medium player hole cards (Width: 46, Height: 64)
fn draw_medium_card(painter: &egui::Painter, pos: egui::Pos2, card: Card) {
    let w = 46.0;
    let h = 64.0;
    let card_rect = egui::Rect::from_center_size(pos, Vec2::new(w, h));

    painter.rect_filled(card_rect, 5.0, Color32::WHITE);
    painter.rect_stroke(card_rect, 5.0, Stroke::new(1.2, Color32::from_rgb(160, 160, 160)));

    let suit_color = match card.suit() {
        Suit::Hearts | Suit::Diamonds => Color32::from_rgb(220, 20, 20),
        Suit::Spades | Suit::Clubs => Color32::from_rgb(18, 18, 18),
    };

    painter.text(
        egui::pos2(pos.x - 12.0, pos.y - 16.0),
        egui::Align2::CENTER_CENTER,
        format!("{}", card.rank().char_code()),
        egui::FontId::proportional(17.0),
        suit_color,
    );

    painter.text(
        egui::pos2(pos.x + 4.0, pos.y + 6.0),
        egui::Align2::CENTER_CENTER,
        card.suit().symbol(),
        egui::FontId::proportional(24.0),
        suit_color,
    );
}

/// Draw face-down card back
fn draw_card_back(painter: &egui::Painter, pos: egui::Pos2) {
    let w = 38.0;
    let h = 54.0;
    let card_rect = egui::Rect::from_center_size(pos, Vec2::new(w, h));

    painter.rect_filled(card_rect, 4.0, Color32::from_rgb(30, 80, 180));
    painter.rect_stroke(card_rect, 4.0, Stroke::new(1.5, Color32::WHITE));

    painter.text(
        pos,
        egui::Align2::CENTER_CENTER,
        "🂠",
        egui::FontId::proportional(22.0),
        Color32::from_rgba_unmultiplied(255, 255, 255, 180),
    );
}

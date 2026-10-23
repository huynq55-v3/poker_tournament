use crate::deep_cfr::{DeepCFRPolicy, DeepCFRTrainer};
use crate::environment::action::{Action, ActionMask, NUM_ACTIONS};
use crate::environment::engine::Environment;
use crate::environment::state::{Position, Street, FEATURE_DIM};
use crate::gui::blind_structure::TournamentBlindSchedule;
use crate::neural_network::MLP;
use crate::poker_core::card::{Card, Suit};
use crate::poker_core::evaluator::evaluate_7_cards;
use crate::self_play::Policy;
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Shape, Stroke, Vec2};
use rand::thread_rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

const ADV_MODEL_PATH: &str = "deep_cfr_model.json";
const STRAT_MODEL_PATH: &str = "deep_cfr_strategy.json";

fn new_untrained_net() -> MLP {
    MLP::new(&[FEATURE_DIM, 256, 256, 128, NUM_ACTIONS])
}

fn load_models() -> (MLP, Option<MLP>, String) {
    let adv = match MLP::load_from_file(ADV_MODEL_PATH) {
        Ok(mlp) if mlp.input_dim() == FEATURE_DIM => mlp,
        _ => new_untrained_net(),
    };
    let strat = MLP::load_from_file(STRAT_MODEL_PATH).ok();
    let status = if strat.is_some() {
        "🟢 Deep CFR Strategy Net + Advantage Net Active".to_string()
    } else {
        "🟡 Advantage Net Active (Strategy Net not found)".to_string()
    };
    (adv, strat, status)
}

fn load_model_from_path(path: &std::path::Path) -> Result<(MLP, Option<MLP>, String), String> {
    match MLP::load_from_file(path) {
        Ok(adv) if adv.input_dim() == FEATURE_DIM => {
            let filename = path.file_name().unwrap_or_default().to_string_lossy();
            let path_str = path.to_string_lossy();
            let strat_opt = if path_str.contains("model") {
                let cand = std::path::PathBuf::from(path_str.replace("model", "strategy"));
                MLP::load_from_file(&cand).ok()
            } else {
                None
            };
            let strat_info = if strat_opt.is_some() { " (+ Strategy Net)" } else { "" };
            Ok((adv, strat_opt, format!("🟢 Loaded: {}{}", filename, strat_info)))
        }
        Ok(adv) => Err(format!(
            "Input dim mismatch: model has {} inputs, expected {}!",
            adv.input_dim(),
            FEATURE_DIM
        )),
        Err(e) => Err(format!("File read error: {}", e)),
    }
}

/// Giao diện giải đấu Poker phong cách PokerNow (Purple Oval Table + Neon Chips + Dynamic Plates)
pub struct PokerNowApp {
    pub env: Environment,
    pub policy: DeepCFRPolicy,
    pub hero_seat: usize,
    pub hand_in_progress: bool,
    pub auto_play_bots: bool,
    pub last_bot_act_time: Instant,
    pub bot_delay_ms: u64,
    pub action_logs: Vec<String>,
    pub model_status: String,
    pub is_training: Arc<AtomicBool>,
    pub train_progress_msg: Arc<RwLock<String>>,
    pub blind_schedule: TournamentBlindSchedule,
    pub num_seats: usize,
    pub selected_num_seats: usize,
    pub button_idx: usize,
    pub hand_number: u32,
    pub tournament_over: bool,
    pub winner_announcement: Option<String>,
    pub eliminated_logged: Vec<bool>,
    pub hero_bet_slider_bb: f32,
    pub show_side_panel: bool,
    pub turn_timer_start: Instant,
    pub last_actor: usize,
}

impl PokerNowApp {
    pub fn new(num_seats: usize) -> Self {
        let seats = num_seats.clamp(2, 8);
        let schedule = TournamentBlindSchedule::new(seats);
        let bl = schedule.current_level();

        let initial_stack = 50 * bl.bb; // 50 BB = 5,000 chips
        let stacks = vec![initial_stack; seats];
        let button_idx = 0;
        let mut env = Environment::new(seats, &stacks, bl.bb, bl.sb, button_idx);
        let mut rng = thread_rng();
        env.reset_hand(&mut rng, button_idx);

        let (adv_net, strat_net, model_status) = load_models();
        let policy = DeepCFRPolicy::with_strategy(Arc::new(RwLock::new(adv_net)), strat_net, true);

        let mut action_logs = Vec::new();
        action_logs.push(format!(
            "🏆 Tournament Started: {} seats, {} chips ({:.0} BB), Blinds: {}/{}",
            seats, initial_stack, initial_stack as f32 / bl.bb as f32, bl.sb, bl.bb
        ));

        let mut app = Self {
            env,
            policy,
            hero_seat: 0,
            hand_in_progress: false,
            auto_play_bots: true,
            last_bot_act_time: Instant::now(),
            bot_delay_ms: 600,
            action_logs,
            model_status,
            is_training: Arc::new(AtomicBool::new(false)),
            train_progress_msg: Arc::new(RwLock::new(String::new())),
            blind_schedule: schedule,
            num_seats: seats,
            selected_num_seats: seats,
            button_idx,
            hand_number: 1,
            tournament_over: false,
            winner_announcement: None,
            eliminated_logged: vec![false; seats],
            hero_bet_slider_bb: 2.0,
            show_side_panel: true,
            turn_timer_start: Instant::now(),
            last_actor: 0,
        };

        app.begin_hand();
        app
    }

    pub fn open_file_and_load_model(&mut self) {
        let file_opt = rfd::FileDialog::new()
            .add_filter("Neural Network Model (*.json)", &["json"])
            .set_title("Select Deep CFR model to load...")
            .pick_file();

        if let Some(path) = file_opt {
            match load_model_from_path(&path) {
                Ok((adv, strat_opt, status_msg)) => {
                    if let Ok(mut net) = self.policy.advantage_net.write() {
                        *net = adv;
                    }
                    if let Ok(mut strat) = self.policy.strategy_net.write() {
                        *strat = strat_opt;
                    }
                    self.model_status = status_msg.clone();
                    self.action_logs.push(format!("Loaded: {:?}", path.file_name().unwrap()));
                }
                Err(err_msg) => {
                    self.model_status = err_msg.clone();
                    self.action_logs.push(format!("Error: {}", err_msg));
                }
            }
        }
    }

    pub fn start_new_tournament(&mut self, seats: usize) {
        self.num_seats = seats.clamp(2, 8);
        self.selected_num_seats = self.num_seats;
        self.blind_schedule = TournamentBlindSchedule::new(self.num_seats);
        let current_blind = self.blind_schedule.current_level();

        let initial_stack = 50 * current_blind.bb;
        let stacks = vec![initial_stack; self.num_seats];
        self.button_idx = 0;
        self.hand_number = 1;
        self.tournament_over = false;
        self.winner_announcement = None;
        self.eliminated_logged = vec![false; self.num_seats];
        self.hero_bet_slider_bb = 2.0;

        self.env = Environment::new(self.num_seats, &stacks, current_blind.bb, current_blind.sb, self.button_idx);
        let mut rng = thread_rng();
        self.env.reset_hand(&mut rng, self.button_idx);

        self.action_logs.clear();
        self.action_logs.push(format!(
            "🏆 Tournament Started: {} seats, {} chips ({:.0} BB), Blinds: {}/{}",
            self.num_seats, initial_stack, initial_stack as f32 / current_blind.bb as f32, current_blind.sb, current_blind.bb
        ));
        self.begin_hand();
    }

    fn surviving_players(&self) -> Vec<usize> {
        (0..self.num_seats)
            .filter(|&i| self.env.state.players[i].stack > 0)
            .collect()
    }

    fn begin_hand(&mut self) {
        self.env.state.players[self.hero_seat].is_bot = false;
        self.hand_in_progress = true;
        self.winner_announcement = None;
        let bl = self.blind_schedule.current_level();
        let surv = self.surviving_players().len();
        self.action_logs.push(format!(
            "--- Hand #{} ({} players left | Level {}: {}/{}) ---",
            self.hand_number, surv, bl.level_num, bl.sb, bl.bb
        ));

        // Log blinds posting
        for i in 0..self.num_seats {
            let p = &self.env.state.players[i];
            if p.current_bet > 0 {
                let name = self.player_name(i);
                let bet_bb = p.current_bet as f32 / bl.bb as f32;
                self.action_logs.push(format!("[Blind] {} posts {} chips ({:.1} BB)", name, p.current_bet, bet_bb));
            }
        }

        if self.env.state.street == Street::Showdown {
            self.hand_in_progress = false;
            self.check_showdown_result();
        }
        self.turn_timer_start = Instant::now();
        self.last_actor = self.env.state.current_player_idx;
    }

    fn start_next_hand(&mut self) {
        let surviving = self.surviving_players();
        if surviving.len() <= 1 {
            self.tournament_over = true;
            self.hand_in_progress = false;
            if let Some(&winner) = surviving.first() {
                let win_name = if winner == self.hero_seat {
                    "Hero (You)".to_string()
                } else {
                    format!("Bot_{}", winner)
                };
                let msg = format!("🏆 TOURNAMENT OVER! {} is Champion!", win_name);
                self.action_logs.push(msg.clone());
                self.winner_announcement = Some(msg);
            }
            return;
        }

        self.hand_number += 1;

        let curr_btn = self.env.state.button_idx;
        let mut next_btn = (curr_btn + 1) % self.num_seats;
        while self.env.state.players[next_btn].stack == 0 {
            next_btn = (next_btn + 1) % self.num_seats;
        }
        self.button_idx = next_btn;

        let leveled_up = self.blind_schedule.on_hand_completed();
        if leveled_up {
            let curr_bl = self.blind_schedule.current_level();
            self.action_logs.push(format!(
                "Level Up: Level {}: {}/{} ({} hands/level)",
                curr_bl.level_num, curr_bl.sb, curr_bl.bb, self.blind_schedule.hands_per_level
            ));
        }

        let curr_bl = self.blind_schedule.current_level();
        self.env.state.bb_size = curr_bl.bb;
        self.env.state.sb_size = curr_bl.sb;
        self.hero_bet_slider_bb = 2.0;

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
        self.action_logs.push(format!("[{:?}] {} -> {}", street, name, action.name()));

        match result {
            Ok(is_hand_over) => {
                if is_hand_over {
                    self.hand_in_progress = false;
                    self.check_showdown_result();
                }
            }
            Err(e) => {
                self.action_logs.push(format!("Error: {}", e));
            }
        }
        self.turn_timer_start = Instant::now();
        self.last_actor = self.env.state.current_player_idx;
    }

    pub fn execute_raise_to(&mut self, target_total: u32) {
        let actor = self.env.state.current_player_idx;
        let name = self.player_name(actor);
        let street = self.env.state.street;
        let bb = self.env.state.bb_size.max(1);
        let target_bb = target_total as f32 / bb as f32;

        let result = self.env.step_raise_to(target_total);
        self.action_logs.push(format!(
            "[{:?}] {} raised to {:.1} BB ({} chips)",
            street, name, target_bb, target_total
        ));

        match result {
            Ok(is_hand_over) => {
                if is_hand_over {
                    self.hand_in_progress = false;
                    self.check_showdown_result();
                }
            }
            Err(e) => {
                self.action_logs.push(format!("Error: {}", e));
            }
        }
        self.turn_timer_start = Instant::now();
        self.last_actor = self.env.state.current_player_idx;
    }

    fn check_showdown_result(&mut self) {
        let remaining: Vec<String> = (0..self.num_seats)
            .filter(|&i| self.env.state.players[i].is_in_hand())
            .map(|i| self.player_name(i))
            .collect();
        let announcement = format!("Hand ended. In hand: {}", remaining.join(", "));
        self.action_logs.push(announcement.clone());

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
                self.action_logs.push(format!("💀 {} is ELIMINATED!", name));
            }
        }

        let surviving = self.surviving_players();
        if surviving.len() <= 1 {
            self.tournament_over = true;
            let winner_idx = surviving.first().copied().unwrap_or(0);
            let win_name = if winner_idx == self.hero_seat {
                "Hero (You)".to_string()
            } else {
                format!("Bot_{}", winner_idx)
            };
            self.winner_announcement = Some(format!("🏆 TOURNAMENT OVER! {} is Champion!", win_name));
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
        self.action_logs.push("Neural Network reloaded.".to_string());
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
            let mut trainer = DeepCFRTrainer::new(FEATURE_DIM, &[256, 256, 128], 30_000, true);
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
                *msg = "Quick training complete. Models hot-reloaded.".to_string();
            }
            is_training_flag.store(false, Ordering::SeqCst);
        });
    }

    /// Đánh giá GTO đầy đủ với Anti-Trash Shove, Postflop Anti-Suicide như `tournament.rs`
    pub fn evaluate_gto_decision(&self, actor: usize) -> ([f32; NUM_ACTIONS], ActionMask, Vec<f32>) {
        let mut mask = self.env.get_action_mask();
        let features = self.env.state.encode_features(actor);
        let to_call = self.env.state.current_highest_bet.saturating_sub(self.env.state.players[actor].current_bet);
        let in_hand_count = self.env.state.players.iter().filter(|p| p.is_in_hand()).count();
        let eff_stack_bb = self.env.state.effective_stack_bb(actor);

        // Preflop Anti-Trash Shove Guard
        let chen_strength = features[166];
        if self.env.state.street == Street::Preflop {
            let min_shove_chen = if eff_stack_bb < 8.0 { 0.20 } else if eff_stack_bb < 15.0 { 0.26 } else { 0.32 };
            if chen_strength < min_shove_chen && mask.is_valid(Action::AllIn) {
                if mask.is_valid(Action::Fold) || mask.is_valid(Action::CheckCall) {
                    mask.0 &= !(1 << Action::AllIn.to_index());
                }
            }
        } else {
            // Postflop Anti-Suicide Guard
            let hand_cat = features[165];
            let has_flush_draw = features[170] >= 0.5;
            let has_straight_draw = features[171] >= 0.5;
            let is_pure_air = hand_cat < 0.15 && !has_flush_draw && !has_straight_draw;

            if to_call > 0 && is_pure_air {
                if mask.is_valid(Action::Fold) || mask.is_valid(Action::CheckCall) {
                    mask.0 &= !((1 << Action::BetSmall.to_index())
                        | (1 << Action::BetPot33.to_index())
                        | (1 << Action::BetPot75.to_index())
                        | (1 << Action::AllIn.to_index()));
                }
            }

            // Multiway Air Guard
            if in_hand_count >= 3 && is_pure_air {
                mask.0 &= !((1 << Action::BetPot75.to_index()) | (1 << Action::AllIn.to_index()));
            }

            // River Anti-Suicide
            if self.env.state.street == Street::River {
                let is_weak_river = hand_cat < 0.15 || (hand_cat < 0.25 && features[166] < 0.20);
                if is_weak_river && mask.is_valid(Action::AllIn) {
                    if mask.is_valid(Action::Fold) || mask.is_valid(Action::CheckCall) {
                        mask.0 &= !(1 << Action::AllIn.to_index());
                    }
                }
            }
        }

        let mut probs = self.policy.get_action_probs(&features, &mask);
        let raw_adv = self.policy.advantage_net.read().unwrap().forward(&features);

        let num_acts = raw_adv.len().min(probs.len());
        if self.policy.strategy_net.read().unwrap().is_none() {
            let mut max_val = f32::NEG_INFINITY;
            for (a_idx, &adv) in raw_adv.iter().enumerate().take(num_acts) {
                if mask.is_index_valid(a_idx) && adv > max_val { max_val = adv; }
            }
            let mut sum_exp = 0.0f32;
            let temp = 0.15;
            for (a_idx, &adv) in raw_adv.iter().enumerate().take(num_acts) {
                if mask.is_index_valid(a_idx) {
                    let e = ((adv - max_val) / temp).exp();
                    probs[a_idx] = e; sum_exp += e;
                } else { probs[a_idx] = 0.0; }
            }
            if sum_exp > 0.0 { for p in &mut probs { *p /= sum_exp; } }
        }

        let mut clean_probs = [0.0f32; NUM_ACTIONS];
        let mut sum_clean = 0.0f32;
        for (i, &p) in probs.iter().enumerate().take(NUM_ACTIONS) {
            if mask.is_index_valid(i) {
                let val = if p > 0.02 { p } else { 0.0 };
                clean_probs[i] = val;
                sum_clean += val;
            }
        }
        if sum_clean <= 1e-4 {
            for (i, &p) in probs.iter().enumerate().take(NUM_ACTIONS) {
                if mask.is_index_valid(i) {
                    clean_probs[i] = p;
                    sum_clean += p;
                }
            }
        }
        if sum_clean > 0.0 {
            for p in &mut clean_probs { *p /= sum_clean; }
        }

        (clean_probs, mask, raw_adv)
    }

    /// Ra quyết định chuẩn PureMaxEv
    pub fn decide_bot_action(&self, actor: usize) -> Action {
        let (clean_probs, mask, raw_adv) = self.evaluate_gto_decision(actor);

        let mut max_gto_p = 0.0f32;
        let mut best_gto_idx = 0;
        for (i, &p) in clean_probs.iter().enumerate().take(NUM_ACTIONS) {
            if mask.is_index_valid(i) && p > max_gto_p {
                max_gto_p = p;
                best_gto_idx = i;
            }
        }

        let min_gto_threshold = 0.08f32;
        let mut candidates = Vec::new();
        for (i, &ev) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
            if !mask.is_index_valid(i) { continue; }
            let p = clean_probs[i];
            if p >= min_gto_threshold && p >= max_gto_p * 0.30 {
                candidates.push((i, ev, p));
            }
        }

        if candidates.is_empty() {
            let ev = raw_adv.get(best_gto_idx).copied().unwrap_or(0.0);
            candidates.push((best_gto_idx, ev, max_gto_p));
        }

        let mut best_i = candidates[0].0;
        let mut best_ev = candidates[0].1;
        for &(idx, ev, _) in &candidates[1..] {
            if ev > best_ev {
                best_i = idx;
                best_ev = ev;
            }
        }

        Action::from_index(best_i).unwrap_or(Action::Fold)
    }
}

impl eframe::App for PokerNowApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(40));

        let curr_actor = self.env.state.current_player_idx;
        if curr_actor != self.last_actor {
            self.turn_timer_start = Instant::now();
            self.last_actor = curr_actor;
        }

        // 1. Bot Auto Moves
        if self.hand_in_progress && !self.tournament_over {
            let is_bot = self.env.state.players[curr_actor].is_bot;
            let is_alive = self.env.state.players[curr_actor].is_active();

            if is_bot && is_alive && self.auto_play_bots
                && self.last_bot_act_time.elapsed() >= Duration::from_millis(self.bot_delay_ms)
            {
                let bot_action = self.decide_bot_action(curr_actor);
                self.execute_action(bot_action);
                self.last_bot_act_time = Instant::now();
            }
        }

        // 2. Top PokerNow Banner
        egui::TopBottomPanel::top("pokernow_top_bar").show(ctx, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("POKER NOW").color(Color32::from_rgb(255, 255, 255)).strong().size(15.0));
                ui.label(RichText::new("+").color(Color32::from_rgb(255, 215, 0)).strong().size(16.0));
                ui.separator();

                ui.label(RichText::new("Seats:").color(Color32::from_rgb(180, 180, 190)));
                if ui.button("-").clicked() && self.selected_num_seats > 2 {
                    self.selected_num_seats -= 1;
                }
                ui.label(RichText::new(format!("{}", self.selected_num_seats)).size(15.0).color(Color32::from_rgb(120, 210, 255)).strong());
                if ui.button("+").clicked() && self.selected_num_seats < 8 {
                    self.selected_num_seats += 1;
                }

                if ui.button(RichText::new("New Tournament").color(Color32::from_rgb(100, 230, 150)).strong()).clicked() {
                    self.start_new_tournament(self.selected_num_seats);
                }

                ui.separator();
                let bl = self.blind_schedule.current_level();
                ui.label(RichText::new(format!("Level {}", bl.level_num)).color(Color32::WHITE).strong());
                ui.label(RichText::new(format!("Blinds: {}/{}", bl.sb, bl.bb)).color(Color32::from_rgb(120, 245, 120)).strong());

                let progress = self.blind_schedule.formatted_hand_progress();
                ui.label(RichText::new(progress).color(Color32::from_rgb(255, 180, 70)));

                let alive_count = self.surviving_players().len();
                ui.label(RichText::new(format!("(Alive: {}/{})", alive_count, self.num_seats)).color(Color32::from_rgb(180, 180, 180)));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(if self.show_side_panel { "Hide GTO Panel" } else { "Show GTO Panel" }).clicked() {
                        self.show_side_panel = !self.show_side_panel;
                    }

                    if ui.button("Next Blind").clicked() && self.blind_schedule.advance_level_manually() {
                        let next_bl = self.blind_schedule.current_level();
                        self.action_logs.push(format!("Blinds -> Level {}: {}/{}", next_bl.level_num, next_bl.sb, next_bl.bb));
                    }

                    let next_bl = self.blind_schedule.current_level();
                    ui.label(RichText::new(format!("NEXT BLIND: {}/{} (Hand #{})", next_bl.sb, next_bl.bb, self.hand_number)).color(Color32::from_rgb(255, 110, 90)).size(12.0));
                });
            });
            ui.add_space(3.0);
        });

        // 3. Optional Right Side GTO & Logs Drawer
        if self.show_side_panel {
            egui::SidePanel::right("pokernow_right_drawer")
                .min_width(320.0)
                .show(ctx, |ui| {
                    ui.add_space(6.0);
                    ui.heading("Deep CFR Neural Net");
                    ui.label(RichText::new(&self.model_status).size(12.0));

                    ui.horizontal(|ui| {
                        if self.is_training.load(Ordering::Relaxed) {
                            ui.add(egui::Spinner::new());
                            if let Ok(msg) = self.train_progress_msg.read() {
                                ui.label(RichText::new(&*msg).color(Color32::from_rgb(255, 200, 60)).size(11.0));
                            }
                        } else {
                            if ui.button("Quick Train (10 Iter)").clicked() {
                                self.start_background_training();
                            }
                            if ui.button("Load Model...").clicked() {
                                self.open_file_and_load_model();
                            }
                            if ui.button("Reload").clicked() {
                                self.reload_model();
                            }
                        }
                    });

                    ui.add_space(10.0);
                    ui.heading("Real-time GTO Probs");
                    ui.separator();

                    let actor = self.env.state.current_player_idx;
                    let (probs, mask, _) = self.evaluate_gto_decision(actor);

                    for a_idx in 0..NUM_ACTIONS {
                        let act = Action::from_index(a_idx).unwrap();
                        let p = probs[a_idx];
                        let is_valid = mask.is_index_valid(a_idx);

                        ui.horizontal(|ui| {
                            let text_color = if is_valid { Color32::WHITE } else { Color32::DARK_GRAY };
                            ui.label(RichText::new(format!("{:12}", act.name())).color(text_color).size(12.5));
                            let bar_color = if is_valid { Color32::from_rgb(70, 170, 245) } else { Color32::GRAY };
                            ui.add(egui::ProgressBar::new(p).text(format!("{:.1}%", p * 100.0)).fill(bar_color));
                        });
                    }

                    ui.add_space(10.0);
                    ui.heading("Tournament Logs");
                    ui.separator();

                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for log in &self.action_logs {
                                ui.label(RichText::new(log).size(11.5));
                            }
                        });
                });
        }

        // 4. Central Canvas: PokerNow Purple Oval Table
        egui::CentralPanel::default().show(ctx, |ui| {
            let table_rect = ui.available_rect_before_wrap();
            let table_height = (table_rect.height() - 170.0).max(320.0);
            let center = Pos2::new(table_rect.center().x, table_rect.min.y + table_height * 0.49);

            let painter = ui.painter();
            let bb = self.env.state.bb_size.max(1);

            // 4.1 Bàn nỉ tím Oval (PokerNow Signature Purple Felt)
            let felt_size = Vec2::new(table_rect.width() * 0.88, table_height * 0.82);
            let felt_rect = Rect::from_center_size(center, felt_size);

            // Viền đệm ngoài (Padded Rail - Dark Charcoal)
            let rail_rect = Rect::from_center_size(center, felt_size + Vec2::new(26.0, 26.0));
            painter.rect_filled(rail_rect, 130.0, Color32::from_rgb(26, 20, 34));
            painter.rect_stroke(rail_rect, 130.0, Stroke::new(3.0, Color32::from_rgb(50, 42, 64)));

            // Lớp nỉ tím sang trọng bên trong
            painter.rect_filled(felt_rect, 120.0, Color32::from_rgb(86, 26, 116));
            // Viền nẹp vàng/bạc mỏng sát mép nỉ
            painter.rect_stroke(felt_rect, 120.0, Stroke::new(2.5, Color32::from_rgb(130, 46, 170)));

            // Watermark chìm ở giữa bàn phong cách PokerNow
            painter.text(
                Pos2::new(center.x, center.y + 46.0),
                egui::Align2::CENTER_CENTER,
                "POWERED BY POKER NOW",
                FontId::proportional(12.5),
                Color32::from_rgba_unmultiplied(180, 110, 220, 40),
            );
            painter.text(
                Pos2::new(center.x, center.y - 12.0),
                egui::Align2::CENTER_CENTER,
                "No Limit Texas Hold'em",
                FontId::proportional(14.0),
                Color32::from_rgba_unmultiplied(180, 110, 220, 35),
            );

            // 4.2 Tổng Pot giữa bàn
            let pot_bb = self.env.state.pot as f32 / bb as f32;
            let pot_str = format!("{}  ({:.1} BB)", self.env.state.pot, pot_bb);
            let pot_pos = Pos2::new(center.x, center.y - 48.0);

            let pot_badge_rect = Rect::from_center_size(pot_pos, Vec2::new(140.0, 26.0));
            painter.rect_filled(pot_badge_rect, 13.0, Color32::from_rgba_unmultiplied(24, 12, 38, 200));
            painter.text(
                pot_pos,
                egui::Align2::CENTER_CENTER,
                pot_str,
                FontId::proportional(17.0),
                Color32::WHITE,
            );

            // 4.3 Bài chung (Community cards) giữa bàn
            let num_comm = self.env.state.community_cards.len();
            let comm_spacing = 74.0;
            let comm_start_x = center.x - ((num_comm as f32 - 1.0).max(0.0) * comm_spacing * 0.5);

            for (c_idx, &card) in self.env.state.community_cards.iter().enumerate() {
                let card_pos = Pos2::new(comm_start_x + (c_idx as f32 * comm_spacing), center.y + 10.0);
                draw_pokernow_large_card(painter, card_pos, card);
            }

            // 4.4 Các vị trí ghế người chơi quanh bàn Oval
            let num_p = self.num_seats;
            let radius_x = felt_size.x * 0.44;
            let radius_y = felt_size.y * 0.41;
            let engine_button = self.env.state.button_idx;

            let alive_seats: Vec<usize> = self.env.state
                .players
                .iter()
                .enumerate()
                .filter(|(_, p)| p.stack > 0 || p.is_all_in)
                .map(|(idx, _)| idx)
                .collect();

            for i in 0..num_p {
                let angle = std::f32::consts::PI * 0.5 + (i as f32 * 2.0 * std::f32::consts::PI / num_p as f32);
                let seat_pos = Pos2::new(center.x + radius_x * angle.cos(), center.y + radius_y * angle.sin());

                let p = &self.env.state.players[i];
                let is_actor = i == self.env.state.current_player_idx && self.hand_in_progress && !self.tournament_over;
                let is_hero = i == self.hero_seat;
                let is_busted = p.stack == 0 && (!self.hand_in_progress || p.is_folded);
                let pos_name = Position::for_seat_alive(i, engine_button, &alive_seats);

                // Điểm đặt Dealer button (D) sát mép bàn phía người chơi (không đè lên lá bài hay plate)
                if i == engine_button && !is_busted {
                    let d_pos = Pos2::new(
                        center.x + (radius_x * 0.76) * angle.cos() + 38.0 * angle.sin(),
                        center.y + (radius_y * 0.74) * angle.sin() - 38.0 * angle.cos(),
                    );
                    draw_dealer_button(painter, d_pos);
                }

                // Vẽ ghế người chơi phong cách PokerNow
                draw_pokernow_player_seat(
                    painter,
                    seat_pos,
                    p,
                    is_hero,
                    is_actor,
                    is_busted,
                    pos_name,
                    bb,
                    self.hand_in_progress,
                    self.env.state.street == Street::Showdown,
                    &self.env.state.community_cards,
                    self.turn_timer_start,
                );

                // Chip cược tròn màu vàng chanh/neon (Neon Lime Chip) in hẳn trong bàn chơi
                if p.current_bet > 0 {
                    let bet_bb = p.current_bet as f32 / bb as f32;
                    let bet_pos = Pos2::new(
                        center.x + (radius_x * 0.60) * angle.cos(),
                        center.y + (radius_y * 0.56) * angle.sin(),
                    );
                    draw_pokernow_bet_chip(painter, bet_pos, p.current_bet, bet_bb);
                }
            }

            // 5. Thanh điều khiển hành động của Hero (Bottom Action Dock)
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
                        ui.label(RichText::new(ann).color(Color32::from_rgb(255, 215, 0)).size(16.0).strong());
                    }
                    if ui.button(RichText::new("Start New Tournament").size(15.0).color(Color32::GREEN).strong()).clicked() {
                        self.start_new_tournament(self.selected_num_seats);
                    }
                } else if !self.hand_in_progress {
                    if ui.button(RichText::new("Deal Next Hand").size(16.0).strong().color(Color32::GREEN)).clicked() {
                        self.start_next_hand();
                    }
                    if let Some(ann) = &self.winner_announcement {
                        ui.label(RichText::new(ann).color(Color32::from_rgb(255, 215, 0)).size(14.0).strong());
                    }
                    if !hero_alive {
                        ui.label(RichText::new("You have busted! Spectating bots or click 'New Tournament' above.").color(Color32::from_rgb(255, 120, 120)));
                    }
                } else if is_hero_turn {
                    let hero = &self.env.state.players[self.hero_seat];
                    let hero_total = hero.current_bet + hero.stack;
                    let to_call = self.env.state.current_highest_bet.saturating_sub(hero.current_bet);
                    let to_call_bb = to_call as f32 / bb as f32;

                    ui.label(RichText::new("YOUR TURN:").size(15.0).color(Color32::from_rgb(255, 225, 90)).strong());

                    if ui.add_enabled(mask.is_valid(Action::Fold), egui::Button::new(RichText::new("Fold").size(14.0))).clicked() {
                        self.execute_action(Action::Fold);
                    }

                    let call_label = if to_call == 0 {
                        "Check".to_string()
                    } else {
                        format!("Call {:.1} BB ({} chips)", to_call_bb, to_call)
                    };
                    if ui.add_enabled(mask.is_valid(Action::CheckCall), egui::Button::new(RichText::new(call_label).size(14.0))).clicked() {
                        self.execute_action(Action::CheckCall);
                    }

                    let b15_target = self.env.raise_target(self.hero_seat, 0.15).min(hero_total);
                    let b15_bb = b15_target as f32 / bb as f32;
                    let b15_label = format!("Bet 15% ({:.1} BB)", b15_bb);
                    if ui.add_enabled(mask.is_valid(Action::BetSmall), egui::Button::new(RichText::new(b15_label).size(14.0))).clicked() {
                        self.execute_action(Action::BetSmall);
                    }

                    let b33_target = self.env.raise_target(self.hero_seat, 0.33).min(hero_total);
                    let b33_bb = b33_target as f32 / bb as f32;
                    let b33_label = format!("Bet 33% ({:.1} BB)", b33_bb);
                    if ui.add_enabled(mask.is_valid(Action::BetPot33), egui::Button::new(RichText::new(b33_label).size(14.0))).clicked() {
                        self.execute_action(Action::BetPot33);
                    }

                    let b75_target = self.env.raise_target(self.hero_seat, 0.75).min(hero_total);
                    let b75_bb = b75_target as f32 / bb as f32;
                    let b75_label = format!("Bet 75% ({:.1} BB)", b75_bb);
                    if ui.add_enabled(mask.is_valid(Action::BetPot75), egui::Button::new(RichText::new(b75_label).size(14.0))).clicked() {
                        self.execute_action(Action::BetPot75);
                    }

                    let allin_bb = hero_total as f32 / bb as f32;
                    let allin_label = format!("All-In ({:.1} BB)", allin_bb);
                    if ui.add_enabled(mask.is_valid(Action::AllIn), egui::Button::new(RichText::new(allin_label).size(14.0).color(Color32::RED).strong())).clicked() {
                        self.execute_action(Action::AllIn);
                    }
                } else {
                    let actor = self.env.state.current_player_idx;
                    ui.label(RichText::new(format!("Bot_{} is thinking...", actor)).size(15.0));

                    if !self.auto_play_bots && ui.button("Step Bot").clicked() {
                        let curr_actor = self.env.state.current_player_idx;
                        let bot_action = self.decide_bot_action(curr_actor);
                        self.execute_action(bot_action);
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut self.auto_play_bots, "Auto Bot Moves");
                    ui.add(egui::Slider::new(&mut self.bot_delay_ms, 100..=1500).text("Bot Delay (ms)"));
                });
            });

            // Row 2: Thanh trượt BB Slider & Presets
            if is_hero_turn {
                let hero = &self.env.state.players[self.hero_seat];
                let hero_total = hero.current_bet + hero.stack;
                let to_call = self.env.state.current_highest_bet.saturating_sub(hero.current_bet);

                let opponents_can_respond = self.env.state.players.iter().enumerate().any(|(i, opp)| i != self.hero_seat && opp.is_active());
                let can_raise = hero.stack > to_call && opponents_can_respond;
                let min_raise_total = self.env.state.current_highest_bet + self.env.state.min_raise;

                if can_raise && hero_total >= min_raise_total {
                    let min_raise_bb = min_raise_total as f32 / bb as f32;
                    let max_raise_bb = hero_total as f32 / bb as f32;

                    if self.hero_bet_slider_bb < min_raise_bb || self.hero_bet_slider_bb > max_raise_bb {
                        self.hero_bet_slider_bb = min_raise_bb;
                    }

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("BB Sizing:").size(14.0).color(Color32::from_rgb(130, 220, 255)).strong());

                        ui.add(
                            egui::Slider::new(&mut self.hero_bet_slider_bb, min_raise_bb..=max_raise_bb)
                                .step_by(0.5)
                                .suffix(" BB")
                        );

                        if ui.button("Min").clicked() { self.hero_bet_slider_bb = min_raise_bb; }
                        if 2.5 >= min_raise_bb && 2.5 <= max_raise_bb {
                            if ui.button("2.5 BB").clicked() { self.hero_bet_slider_bb = 2.5; }
                        }
                        if 3.0 >= min_raise_bb && 3.0 <= max_raise_bb {
                            if ui.button("3.0 BB").clicked() { self.hero_bet_slider_bb = 3.0; }
                        }
                        if 5.0 >= min_raise_bb && 5.0 <= max_raise_bb {
                            if ui.button("5.0 BB").clicked() { self.hero_bet_slider_bb = 5.0; }
                        }
                        if 10.0 >= min_raise_bb && 10.0 <= max_raise_bb {
                            if ui.button("10.0 BB").clicked() { self.hero_bet_slider_bb = 10.0; }
                        }
                        if ui.button("Max").clicked() { self.hero_bet_slider_bb = max_raise_bb; }

                        let target_chips = ((self.hero_bet_slider_bb * bb as f32).round() as u32).clamp(min_raise_total, hero_total);
                        let is_shove = target_chips >= hero_total;
                        let btn_text = if is_shove {
                            format!("All-In ({:.1} BB | {} chips)", max_raise_bb, hero_total)
                        } else {
                            format!("Raise {:.1} BB ({} chips)", self.hero_bet_slider_bb, target_chips)
                        };

                        if ui.button(RichText::new(btn_text).size(14.0).strong().color(Color32::from_rgb(255, 215, 0))).clicked() {
                            self.execute_raise_to(target_chips);
                        }
                    });
                }
            }
        });
    }
}

/// Vẽ một ghế ngồi chuẩn phong cách PokerNow (Cards + Name/Stack plate + Spotlight khi tới lượt)
fn draw_pokernow_player_seat(
    painter: &egui::Painter,
    pos: Pos2,
    player: &crate::environment::state::PlayerState,
    is_hero: bool,
    is_actor: bool,
    is_busted: bool,
    pos_name: Position,
    bb: u32,
    hand_in_progress: bool,
    is_showdown: bool,
    community_cards: &[Card],
    turn_timer_start: Instant,
) {
    if is_busted {
        // Người chơi đã bị loại
        let plate_rect = Rect::from_center_size(pos, Vec2::new(110.0, 38.0));
        painter.rect_filled(plate_rect, 6.0, Color32::from_rgba_unmultiplied(20, 18, 24, 210));
        painter.rect_stroke(plate_rect, 6.0, Stroke::new(1.0, Color32::from_rgb(60, 40, 40)));

        let p_name = if is_hero { "Hero (You)".to_string() } else { format!("Bot_{}", player.seat_idx) };
        painter.text(Pos2::new(pos.x, pos.y - 7.0), egui::Align2::CENTER_CENTER, p_name, FontId::proportional(12.0), Color32::from_rgb(140, 70, 70));
        painter.text(Pos2::new(pos.x, pos.y + 8.0), egui::Align2::CENTER_CENTER, "OUT", FontId::proportional(13.0), Color32::from_rgb(200, 60, 60));
        return;
    }

    if player.is_folded {
        // Phong cách PokerNow khi FOLD: Không có bài, hiển thị khối xám "X FOLD" + Tên/Stack mờ
        let seat_width = 136.0;
        let seat_height = 42.0;
        let fold_rect = Rect::from_center_size(pos, Vec2::new(seat_width, seat_height));

        painter.rect_filled(fold_rect, 6.0, Color32::from_rgba_unmultiplied(26, 28, 35, 210));
        painter.rect_stroke(fold_rect, 6.0, Stroke::new(1.0, Color32::from_rgb(50, 55, 65)));

        // Biểu tượng X FOLD bên trái (ASCII 'X' chuẩn, không lo lỗi font Linux)
        let x_box = Rect::from_center_size(Pos2::new(pos.x - 44.0, pos.y), Vec2::new(32.0, 32.0));
        painter.rect_filled(x_box, 4.0, Color32::from_rgba_unmultiplied(40, 44, 54, 230));
        painter.text(Pos2::new(x_box.center().x, x_box.center().y - 4.0), egui::Align2::CENTER_CENTER, "X", FontId::proportional(13.0), Color32::from_rgb(120, 125, 135));
        painter.text(Pos2::new(x_box.center().x, x_box.center().y + 8.0), egui::Align2::CENTER_CENTER, "FOLD", FontId::proportional(8.5), Color32::from_rgb(120, 125, 135));

        // Tên & Stack bên phải
        let p_name = if is_hero { "Hero".to_string() } else { format!("Bot_{}", player.seat_idx) };
        painter.text(Pos2::new(pos.x + 8.0, pos.y - 7.0), egui::Align2::CENTER_CENTER, format!("{} ({:?})", p_name, pos_name), FontId::proportional(12.5), Color32::from_rgb(150, 155, 165));
        let stack_bb = player.stack as f32 / bb as f32;
        painter.text(Pos2::new(pos.x + 8.0, pos.y + 8.0), egui::Align2::CENTER_CENTER, format!("{} ({:.1} BB)", player.stack, stack_bb), FontId::proportional(11.5), Color32::from_rgb(110, 115, 125));
        return;
    }

    // NGƯỜI CHƠI ĐANG THAM GIA VÁN BÀI:
    let (cards_center, plate_center, card_spacing, badge_offset_y, badge_width) = if is_hero {
        (
            Pos2::new(pos.x - 40.0, pos.y),
            Pos2::new(pos.x + 44.0, pos.y),
            17.0,
            48.0,
            96.0,
        )
    } else {
        (
            Pos2::new(pos.x - 32.0, pos.y),
            Pos2::new(pos.x + 36.0, pos.y),
            13.0,
            42.0,
            82.0,
        )
    };

    let should_reveal_cards = is_hero || !hand_in_progress || is_showdown;

    // 1. Vẽ 2 lá bài tẩy
    if should_reveal_cards {
        if let (Some(c0), Some(c1)) = (player.hole_cards[0], player.hole_cards[1]) {
            // Lá 1 hơi chếch sang trái
            draw_pokernow_hole_card(painter, Pos2::new(cards_center.x - card_spacing, cards_center.y), c0, is_hero);
            // Lá 2 đè nhẹ lên lá 1 sang phải
            draw_pokernow_hole_card(painter, Pos2::new(cards_center.x + card_spacing, cards_center.y), c1, is_hero);
        }
    } else {
        // Bài úp phong cách PokerNow (Lưng tối viền vàng cam)
        draw_pokernow_card_back(painter, Pos2::new(cards_center.x - card_spacing, cards_center.y), is_hero);
        draw_pokernow_card_back(painter, Pos2::new(cards_center.x + card_spacing, cards_center.y), is_hero);
    }

    // 2. Thẻ Name & Stack Plate (Đặc trưng PokerNow!)
    let plate_rect = Rect::from_center_size(plate_center, Vec2::new(98.0, 44.0));

    if is_actor {
        // Khi đến lượt người chơi: Thẻ phát sáng rực rỡ màu trắng (White Spotlight Aura)
        let aura_rect = plate_rect.expand(4.0);
        painter.rect_filled(aura_rect, 8.0, Color32::from_rgba_unmultiplied(255, 255, 255, 60));
        painter.rect_filled(plate_rect, 6.0, Color32::WHITE);
        painter.rect_stroke(plate_rect, 6.0, Stroke::new(2.0, Color32::from_rgb(255, 220, 80)));

        // Tên người chơi màu đen đậm
        let p_name = if is_hero { "Hero".to_string() } else { format!("Bot_{}", player.seat_idx) };
        painter.text(Pos2::new(plate_center.x, plate_center.y - 8.0), egui::Align2::CENTER_CENTER, format!("{} ({:?})", p_name, pos_name), FontId::proportional(12.5), Color32::BLACK);

        // Stack màu đen đậm
        let stack_bb = player.stack as f32 / bb as f32;
        painter.text(Pos2::new(plate_center.x, plate_center.y + 7.0), egui::Align2::CENTER_CENTER, format!("{} ({:.1} BB)", player.stack, stack_bb), FontId::proportional(11.5), Color32::from_rgb(30, 30, 30));

        // Thanh tiến độ suy nghĩ (Thinking Timer Bar) màu tím/xanh neon ở đáy thẻ
        let elapsed_secs = turn_timer_start.elapsed().as_secs_f32();
        let max_secs = 20.0f32;
        let progress = (1.0 - (elapsed_secs / max_secs)).clamp(0.0, 1.0);
        let bar_rect = Rect::from_min_max(
            Pos2::new(plate_rect.min.x + 3.0, plate_rect.max.y - 4.0),
            Pos2::new(plate_rect.min.x + 3.0 + (plate_rect.width() - 6.0) * progress, plate_rect.max.y - 1.0),
        );
        painter.rect_filled(bar_rect, 1.5, Color32::from_rgb(130, 40, 200));
    } else {
        // Trạng thái bình thường: Thẻ tối màu viền xám
        painter.rect_filled(plate_rect, 6.0, Color32::from_rgba_unmultiplied(22, 25, 34, 235));
        painter.rect_stroke(plate_rect, 6.0, Stroke::new(1.2, if is_hero { Color32::from_rgb(90, 170, 250) } else { Color32::from_rgb(60, 65, 78) }));

        let p_name = if is_hero { "Hero".to_string() } else { format!("Bot_{}", player.seat_idx) };
        painter.text(Pos2::new(plate_center.x, plate_center.y - 8.0), egui::Align2::CENTER_CENTER, format!("{} ({:?})", p_name, pos_name), FontId::proportional(12.5), if is_hero { Color32::from_rgb(130, 210, 255) } else { Color32::WHITE });

        let stack_bb = player.stack as f32 / bb as f32;
        painter.text(Pos2::new(plate_center.x, plate_center.y + 8.0), egui::Align2::CENTER_CENTER, format!("{} ({:.1} BB)", player.stack, stack_bb), FontId::proportional(12.0), Color32::from_rgb(130, 240, 130));
    }

    if player.is_all_in {
        let allin_badge = Rect::from_center_size(Pos2::new(plate_center.x, plate_center.y + 26.0), Vec2::new(60.0, 16.0));
        painter.rect_filled(allin_badge, 3.0, Color32::from_rgb(220, 30, 30));
        painter.text(Pos2::new(plate_center.x, plate_center.y + 26.0), egui::Align2::CENTER_CENTER, "ALL-IN", FontId::proportional(10.5), Color32::WHITE);
    }

    // 3. Badge chỉ số liên kết bài (HIGH CARD, TWO PAIR, FLUSH...) ngay phía dưới lá bài (y hệt ảnh PokerNow)
    let show_hand_rank = (is_hero || !hand_in_progress || is_showdown) && community_cards.len() >= 3;
    if show_hand_rank {
        if let (Some(c0), Some(c1)) = (player.hole_cards[0], player.hole_cards[1]) {
            let mut all_cards = Vec::with_capacity(7);
            all_cards.push(c0);
            all_cards.push(c1);
            all_cards.extend_from_slice(community_cards);
            let rank = evaluate_7_cards(&all_cards);
            let cat_name = rank.category().name().to_uppercase();

            let badge_pos = Pos2::new(cards_center.x, cards_center.y + badge_offset_y);
            let badge_rect = Rect::from_center_size(badge_pos, Vec2::new(badge_width, 18.0));

            // Badge đỏ bo tròn chữ trắng đậm (Chuẩn PokerNow!)
            painter.rect_filled(badge_rect, 4.0, Color32::from_rgb(220, 40, 40));
            painter.rect_stroke(badge_rect, 4.0, Stroke::new(1.0, Color32::from_rgb(255, 100, 100)));
            painter.text(badge_pos, egui::Align2::CENTER_CENTER, cat_name, FontId::proportional(10.0), Color32::WHITE);
        }
    }
}

/// Nút Dealer Button (D) tròn trắng viền đen
fn draw_dealer_button(painter: &egui::Painter, pos: Pos2) {
    let r = 12.0;
    // Đổ bóng nhẹ
    painter.circle_filled(Pos2::new(pos.x + 1.0, pos.y + 1.5), r, Color32::from_rgba_unmultiplied(0, 0, 0, 120));
    painter.circle_filled(pos, r, Color32::WHITE);
    painter.circle_stroke(pos, r, Stroke::new(1.5, Color32::from_rgb(80, 80, 90)));
    painter.text(pos, egui::Align2::CENTER_CENTER, "D", FontId::proportional(13.5), Color32::from_rgb(20, 30, 80));
}

/// Viên chip cược tròn màu vàng chanh/neon (Neon Lime Chip) in thẳng trên bàn chơi
fn draw_pokernow_bet_chip(painter: &egui::Painter, pos: Pos2, chips: u32, bb: f32) {
    let text = if bb >= 1.0 {
        format!("{:.1} BB ({})", bb, chips)
    } else {
        format!("{}", chips)
    };

    let chip_width = (text.len() as f32 * 7.5 + 20.0).max(42.0);
    let chip_rect = Rect::from_center_size(pos, Vec2::new(chip_width, 22.0));

    // Đổ bóng dưới chip
    let shadow_rect = Rect::from_center_size(Pos2::new(pos.x + 1.0, pos.y + 1.5), Vec2::new(chip_width, 22.0));
    painter.rect_filled(shadow_rect, 11.0, Color32::from_rgba_unmultiplied(0, 0, 0, 130));

    // Chip màu vàng chanh/neon (`#D4FF32`) chuẩn PokerNow
    painter.rect_filled(chip_rect, 11.0, Color32::from_rgb(212, 255, 50));
    painter.rect_stroke(chip_rect, 11.0, Stroke::new(1.2, Color32::from_rgb(140, 180, 20)));

    // Chữ màu đen đậm
    painter.text(pos, egui::Align2::CENTER_CENTER, text, FontId::proportional(11.5), Color32::BLACK);
}

/// Lá bài chung giữa bàn (Community Card)
fn draw_pokernow_large_card(painter: &egui::Painter, pos: Pos2, card: Card) {
    let w = 58.0;
    let h = 82.0;
    let card_rect = Rect::from_center_size(pos, Vec2::new(w, h));

    // Đổ bóng dưới lá bài
    let shadow_rect = Rect::from_center_size(Pos2::new(pos.x + 2.0, pos.y + 3.0), Vec2::new(w, h));
    painter.rect_filled(shadow_rect, 5.0, Color32::from_rgba_unmultiplied(10, 5, 20, 140));

    // Thân lá bài trắng tinh bo góc
    painter.rect_filled(card_rect, 5.0, Color32::WHITE);
    painter.rect_stroke(card_rect, 5.0, Stroke::new(1.2, Color32::from_rgb(190, 190, 195)));

    let suit_color = match card.suit() {
        Suit::Hearts | Suit::Diamonds => Color32::from_rgb(210, 25, 25),
        Suit::Spades | Suit::Clubs => Color32::from_rgb(25, 25, 30),
    };

    painter.text(
        Pos2::new(pos.x - 14.0, pos.y - 24.0),
        egui::Align2::CENTER_CENTER,
        format!("{}", card.rank().char_code()),
        FontId::proportional(22.0),
        suit_color,
    );

    painter.text(
        Pos2::new(pos.x + 5.0, pos.y + 8.0),
        egui::Align2::CENTER_CENTER,
        card.suit().symbol(),
        FontId::proportional(34.0),
        suit_color,
    );
}

/// Lá bài tẩy của người chơi (Hole card)
fn draw_pokernow_hole_card(painter: &egui::Painter, pos: Pos2, card: Card, is_hero: bool) {
    let (w, h, r_size, s_size, rx, ry, sx, sy) = if is_hero {
        (50.0, 70.0, 20.0, 30.0, -12.0, -19.0, 5.0, 8.0)
    } else {
        (42.0, 60.0, 17.0, 25.0, -10.0, -16.0, 4.0, 7.0)
    };
    let card_rect = Rect::from_center_size(pos, Vec2::new(w, h));

    // Drop shadow
    let shadow_rect = Rect::from_center_size(Pos2::new(pos.x + 1.5, pos.y + 2.0), Vec2::new(w, h));
    painter.rect_filled(shadow_rect, 4.0, Color32::from_rgba_unmultiplied(10, 5, 20, 120));

    painter.rect_filled(card_rect, 4.0, Color32::WHITE);
    painter.rect_stroke(card_rect, 4.0, Stroke::new(1.0, Color32::from_rgb(180, 180, 185)));

    let suit_color = match card.suit() {
        Suit::Hearts | Suit::Diamonds => Color32::from_rgb(210, 25, 25),
        Suit::Spades | Suit::Clubs => Color32::from_rgb(25, 25, 30),
    };

    painter.text(
        Pos2::new(pos.x + rx, pos.y + ry),
        egui::Align2::CENTER_CENTER,
        format!("{}", card.rank().char_code()),
        FontId::proportional(r_size),
        suit_color,
    );

    painter.text(
        Pos2::new(pos.x + sx, pos.y + sy),
        egui::Align2::CENTER_CENTER,
        card.suit().symbol(),
        FontId::proportional(s_size),
        suit_color,
    );
}

/// Lưng bài úp phong cách PokerNow (Nền tối hoa văn vàng cam)
fn draw_pokernow_card_back(painter: &egui::Painter, pos: Pos2, is_hero: bool) {
    let (w, h) = if is_hero { (50.0, 70.0) } else { (42.0, 60.0) };
    let card_rect = Rect::from_center_size(pos, Vec2::new(w, h));

    // Shadow
    let shadow_rect = Rect::from_center_size(Pos2::new(pos.x + 1.5, pos.y + 2.0), Vec2::new(w, h));
    painter.rect_filled(shadow_rect, 4.0, Color32::from_rgba_unmultiplied(10, 5, 20, 120));

    // Nền tối và viền cam/vàng PokerNow
    painter.rect_filled(card_rect, 4.0, Color32::from_rgb(28, 20, 36));
    painter.rect_stroke(card_rect, 4.0, Stroke::new(1.4, Color32::from_rgb(200, 130, 45)));

    // Họa tiết khung trong
    let inner_rect = card_rect.shrink(3.0);
    painter.rect_stroke(inner_rect, 2.5, Stroke::new(0.8, Color32::from_rgb(160, 95, 30)));

    // Diamond pattern
    let d_size = 7.0;
    let diamond_pts = vec![
        Pos2::new(pos.x, pos.y - d_size),
        Pos2::new(pos.x + d_size * 0.7, pos.y),
        Pos2::new(pos.x, pos.y + d_size),
        Pos2::new(pos.x - d_size * 0.7, pos.y),
    ];
    painter.add(Shape::convex_polygon(
        diamond_pts,
        Color32::from_rgb(215, 140, 50),
        Stroke::new(0.8, Color32::from_rgb(240, 180, 80)),
    ));
}

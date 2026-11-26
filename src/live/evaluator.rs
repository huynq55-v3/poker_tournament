use crate::deep_cfr::{DeepCFRPolicy, SearchTrigger, SubgameResult, SubgameSolver};
use crate::environment::{Action, ActionMask, NUM_ACTIONS};
use crate::live::os_input::execute_bot_action;
use crate::live::protocol::{build_environment, HandState};
use crate::live::{DecisionMode, InPageAction, PlayConfig};
use crate::poker_core::Card;
use crate::self_play::Policy;
use std::thread;

fn blueprint_eval(
    policy: &DeepCFRPolicy,
    features: &[f32],
    mask: &ActionMask,
) -> ([f32; NUM_ACTIONS], [f32; NUM_ACTIONS], Option<SubgameResult>) {
    let mut probs = policy.get_action_probs(features, mask);
    let raw_vec = policy.advantage_net.read().unwrap().forward(features);

    let num_acts = raw_vec.len().min(probs.len());
    if policy.strategy_net.read().unwrap().is_none() {
        let mut max_val = f32::NEG_INFINITY;
        for (a_idx, &adv) in raw_vec.iter().enumerate().take(num_acts) {
            if mask.is_index_valid(a_idx) && adv > max_val {
                max_val = adv;
            }
        }
        let mut sum_exp = 0.0f32;
        let temp = 0.15;
        for (a_idx, &adv) in raw_vec.iter().enumerate().take(num_acts) {
            if mask.is_index_valid(a_idx) {
                let e = ((adv - max_val) / temp).exp();
                probs[a_idx] = e;
                sum_exp += e;
            } else {
                probs[a_idx] = 0.0;
            }
        }
        if sum_exp > 0.0 {
            for p in &mut probs {
                *p /= sum_exp;
            }
        }
    }

    let mut raw_adv = [0.0f32; NUM_ACTIONS];
    for (i, &v) in raw_vec.iter().enumerate().take(NUM_ACTIONS) {
        raw_adv[i] = v;
    }
    (raw_adv, probs, None)
}

pub fn run_gto_evaluation(
    state: &HandState,
    hole: [Card; 2],
    policy: &DeepCFRPolicy,
    cfg: &PlayConfig,
) -> Option<InPageAction> {
    let ctx = build_environment(state, hole, cfg.mode)?;
    let env = &ctx.env;
    let num_players = ctx.active_seats.len();
    let hero_seat = ctx.hero_seat;
    let actual_highest_bet = ctx.actual_highest_bet;
    let true_table_pot = ctx.true_table_pot;
    let safe_min_raise = ctx.safe_min_raise;
    let to_call = ctx.to_call;
    let max_total_bet = ctx.max_total_bet;
    let can_raise = ctx.can_raise;
    let in_hand_count = ctx.in_hand_count;
    let hero_stack_behind = env.state.players[hero_seat].stack;

    let mut mask = env.get_action_mask();
    if !can_raise {
        mask.0 &= !((1 << Action::BetSmall.to_index())
            | (1 << Action::BetPot33.to_index())
            | (1 << Action::BetPot75.to_index())
            | (1 << Action::AllIn.to_index()));
    }

    let features = env.state.encode_features(hero_seat);

    let mut solver_cfg = cfg.solver_config.clone();
    solver_cfg.use_icm = !cfg.mode.is_cash_game();
    let solver = SubgameSolver::new(solver_cfg);

    let (raw_adv, probs, solver_result_opt) = if cfg.enable_solver {
        let trigger = solver.should_trigger_search(env, hero_seat, cfg.force_search);
        if trigger != SearchTrigger::FastPolicyFallback {
            let res = solver.resolve(
                env,
                hero_seat,
                &policy.advantage_net.read().unwrap(),
                policy.strategy_net.read().unwrap().as_ref(),
                trigger,
                Some(mask),
            );
            (res.action_evs, res.action_probs, Some(res))
        } else {
            blueprint_eval(policy, &features, &mask)
        }
    } else {
        blueprint_eval(policy, &features, &mask)
    };

    println!("\n============================================================");
    println!("🎯 [{} - ĐẾN LƯỢT BẠN] - VÒNG: {:?}", cfg.mode.label(), state.street);
    println!("👥 Người còn trong ván: {} / Ghế active trên bàn: {}", in_hand_count, num_players);
    println!("🃏 Bài tẩy : [ {} {} ]", hole[0], hole[1]);
    if !state.community_cards.is_empty() {
        let board_str: Vec<String> = state.community_cards.iter().map(|c| format!("{}", c)).collect();
        println!("🎴 Bài chung: [ {} ]", board_str.join(" "));
    }
    let call_status = if to_call == 0 { "(Đang check)" } else { "" };
    println!(
        "💰 Tổng Pot thực tế: {} chip | Cần Call: {} chip {} | Blind: {}/{} chip",
        true_table_pot, to_call, call_status, state.sb_size, state.bb_size
    );
    let eff_bb = env.state.effective_stack_bb(hero_seat);
    let eff_chips = (eff_bb * state.bb_size.max(1) as f32).round() as u32;
    println!("⚡ Effective Stack : {} chip ({:.1} BB)", eff_chips, eff_bb);

    let ev_unit = if solver.config.use_icm { "ICM" } else { "BB" };
    if let Some(ref res) = solver_result_opt {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║ 🔍 [SUBGAME SOLVER ACTIVATED - {}]                         ║", ev_unit);
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("🚨 Lý do kích hoạt : {}", res.trigger);
        println!(
            "⏱️  Thời gian giải   : {:.2} ms ({} iters, max depth {})",
            res.duration_ms, res.iterations_run, solver.config.max_depth
        );
    } else if cfg.enable_solver {
        println!("⚡ [FAST POLICY INFERENCE] Tình huống chuẩn Blueprint -> Mạng Nơ-ron tức thì");
    } else {
        println!("⚡ [BLUEPRINT NET ONLY] Subgame Solver đang tắt (--no-solver)");
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
        for p in &mut clean_probs {
            *p /= sum_clean;
        }
    }

    let (chosen_idx, decision_note) = match cfg.decision_mode {
        DecisionMode::PureMaxEv => {
            if let Some(ref res) = solver_result_opt {
                let mut best_i = 0;
                let mut best_ev = f32::NEG_INFINITY;
                let mut best_p = -1.0f32;
                for (i, &ev) in res.action_evs.iter().enumerate().take(NUM_ACTIONS) {
                    if mask.is_index_valid(i) {
                        let p = res.action_probs[i];
                        if ev > best_ev || ((ev - best_ev).abs() < 1e-4 && p > best_p) {
                            best_ev = ev;
                            best_p = p;
                            best_i = i;
                        }
                    }
                }
                let best_act = Action::from_index(best_i).unwrap();
                let note = format!(
                    "⚡ [SOLVER EV MAX] Nhánh EV cao nhất từ Search Tree: {:?} (EV: {:>+6.2} {}, GTO: {:.1}%)",
                    best_act,
                    best_ev,
                    ev_unit,
                    best_p * 100.0
                );
                (best_i, note)
            } else {
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
                    if !mask.is_index_valid(i) {
                        continue;
                    }
                    let p = clean_probs[i];
                    if p >= min_gto_threshold && p >= max_gto_p * 0.70 {
                        candidates.push((i, ev, p));
                    }
                }

                if candidates.is_empty() {
                    let ev = raw_adv.get(best_gto_idx).copied().unwrap_or(0.0);
                    candidates.push((best_gto_idx, ev, max_gto_p));
                }

                let mut best_i = candidates[0].0;
                let mut best_ev = candidates[0].1;
                let mut best_p = candidates[0].2;
                for &(idx, ev, p) in &candidates[1..] {
                    if ev > best_ev {
                        best_i = idx;
                        best_ev = ev;
                        best_p = p;
                    }
                }

                let best_act = Action::from_index(best_i).unwrap();
                let note = format!(
                    "⚡ [GTO + EV MAX] Nhánh chọn: {:?} (GTO: {:.1}%, EV: {:>+6.2})",
                    best_act,
                    best_p * 100.0,
                    best_ev
                );
                (best_i, note)
            }
        }
        DecisionMode::PureGtoRoll => {
            let mut rng = rand::thread_rng();
            let idx = if let Ok(dist) = rand::distributions::WeightedIndex::new(clean_probs) {
                use rand::distributions::Distribution;
                dist.sample(&mut rng)
            } else {
                clean_probs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask.is_index_valid(*i))
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };
            (
                idx,
                format!("🎲 [PURE GTO ROLL] Tung xúc xắc theo % lý thuyết ({:.1}%)", clean_probs[idx] * 100.0),
            )
        }
        DecisionMode::GuardedMixing => {
            let mut rng = rand::thread_rng();
            let mut ev_max = f32::NEG_INFINITY;
            let mut best_ev_idx = 0;
            for (i, &ev) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
                if mask.is_index_valid(i) && ev > ev_max {
                    ev_max = ev;
                    best_ev_idx = i;
                }
            }
            let (delta_general, delta_allin, dominance_gap) =
                if solver_result_opt.is_some() && !solver.config.use_icm {
                    let pot_bb = (true_table_pot as f32 / state.bb_size.max(1) as f32).clamp(1.0, 40.0);
                    ((0.05 * pot_bb).max(0.25), (0.02 * pot_bb).max(0.10), (0.08 * pot_bb).max(0.35))
                } else {
                    (0.20, 0.08, 0.35)
                };
            let mut safe_candidates = Vec::new();
            let mut second_best_ev = f32::NEG_INFINITY;

            for (i, &ev) in raw_adv.iter().enumerate().take(NUM_ACTIONS) {
                if !mask.is_index_valid(i) {
                    continue;
                }
                let act = Action::from_index(i).unwrap();
                let allowed_delta = if act == Action::AllIn { delta_allin } else { delta_general };
                if ev < ev_max && ev > second_best_ev {
                    second_best_ev = ev;
                }
                if ev >= ev_max - allowed_delta && clean_probs[i] > 0.06 {
                    safe_candidates.push(i);
                }
            }

            if ev_max - second_best_ev > dominance_gap || safe_candidates.len() <= 1 {
                (best_ev_idx, format!("⚡ [ƯU THẾ ÁP ĐẢO] EV {:+.2} -> Chọn 100%", ev_max))
            } else {
                let mut guarded_probs = [0.0f32; NUM_ACTIONS];
                let mut sum_guarded = 0.0f32;
                for &idx in &safe_candidates {
                    let w = clean_probs[idx];
                    guarded_probs[idx] = w;
                    sum_guarded += w;
                }
                let picked = if sum_guarded > 0.0 {
                    for p in &mut guarded_probs {
                        *p /= sum_guarded;
                    }
                    if let Ok(dist) = rand::distributions::WeightedIndex::new(guarded_probs) {
                        use rand::distributions::Distribution;
                        dist.sample(&mut rng)
                    } else {
                        best_ev_idx
                    }
                } else {
                    best_ev_idx
                };
                (
                    picked,
                    format!("🛡️ [GUARDED MIXING] Roll an toàn ({:.1}%)", guarded_probs[picked] * 100.0),
                )
            }
        }
    };

    let get_action_desc = |act: Action| -> String {
        match act {
            Action::Fold => "0".to_string(),
            Action::CheckCall => {
                if to_call == 0 {
                    "Check".to_string()
                } else {
                    format!("Call {}", to_call)
                }
            }
            Action::AllIn => format!("All-In ({})", hero_stack_behind),
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                let frac = match act {
                    Action::BetSmall => 0.15,
                    Action::BetPot33 => 0.33,
                    Action::BetPot75 => 0.75,
                    _ => unreachable!(),
                };
                let pot_after_call = true_table_pot + to_call;
                let raise_by = ((pot_after_call as f32) * frac).round() as u32;
                let target_total = (actual_highest_bet + raise_by.max(safe_min_raise)).min(max_total_bet);
                format!("Tố lên {}", target_total)
            }
        }
    };

    let src_tag = if solver_result_opt.is_some() {
        format!("SOLVER - {}", ev_unit)
    } else {
        "BLUEPRINT NET".to_string()
    };
    println!("📊 BẢNG ĐÁNH GIÁ NƯỚC ĐI [{} | {}]:", cfg.mode.label(), src_tag);
    for (i, &p) in probs.iter().enumerate() {
        let act = Action::from_index(i).unwrap();
        let is_valid = mask.is_index_valid(i);
        let rec = if is_valid && i == chosen_idx {
            match cfg.decision_mode {
                DecisionMode::PureMaxEv => " ⭐ TỐI ĐA EV",
                DecisionMode::GuardedMixing => " ⭐ GUARDED",
                DecisionMode::PureGtoRoll => " ⭐ ROLL",
            }
        } else {
            ""
        };
        let status = if is_valid { "" } else { " (Bị khóa)" };
        let raw_val = raw_adv.get(i).copied().unwrap_or(0.0);
        println!(
            "    {:<14}: EV: {:>+6.2} (GTO: {:>5.1}%) | [{:<15}]{}{}",
            act.name(),
            raw_val,
            p * 100.0,
            get_action_desc(act),
            rec,
            status
        );
    }

    if let Some(act) = Action::from_index(chosen_idx) {
        println!("{}", decision_note);
        println!("👉 Quyết định cuối cùng: {:?}", act);
    }
    println!("============================================================\n");

    if cfg.advice_only {
        println!("👁️ [ADVICE ONLY] Đã hiển thị gợi ý, không tự động bấm nút.");
        return None;
    }

    if let Some(best_action) = Action::from_index(chosen_idx) {
        let mut target_total = actual_highest_bet;
        match best_action {
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                let frac = match best_action {
                    Action::BetSmall => 0.15,
                    Action::BetPot33 => 0.33,
                    Action::BetPot75 => 0.75,
                    _ => unreachable!(),
                };
                let pot_after_call = true_table_pot + to_call;
                let raise_by = ((pot_after_call as f32) * frac).round() as u32;
                target_total = (actual_highest_bet + raise_by.max(safe_min_raise)).min(max_total_bet);
            }
            Action::AllIn => {
                target_total = max_total_bet;
            }
            _ => {}
        }

        let safe_action = if !can_raise && best_action != Action::Fold && best_action != Action::CheckCall {
            Action::CheckCall
        } else {
            best_action
        };

        let in_page_cmd = match safe_action {
            Action::Fold => InPageAction {
                action: "fold".to_string(),
                amount: 0,
                to_call: 0,
            },
            Action::CheckCall => {
                if to_call == 0 {
                    InPageAction {
                        action: "check".to_string(),
                        amount: 0,
                        to_call: 0,
                    }
                } else {
                    InPageAction {
                        action: "call".to_string(),
                        amount: to_call,
                        to_call,
                    }
                }
            }
            Action::AllIn => InPageAction {
                action: "all_in".to_string(),
                amount: max_total_bet,
                to_call,
            },
            Action::BetSmall | Action::BetPot33 | Action::BetPot75 => {
                let final_bet = target_total.min(max_total_bet);
                if final_bet >= max_total_bet {
                    InPageAction {
                        action: "all_in".to_string(),
                        amount: max_total_bet,
                        to_call,
                    }
                } else {
                    InPageAction {
                        action: "raise".to_string(),
                        amount: final_bet,
                        to_call,
                    }
                }
            }
        };

        if cfg.use_os_enigo {
            thread::spawn(move || {
                execute_bot_action(safe_action, to_call, target_total, max_total_bet, can_raise);
            });
        } else {
            println!(
                "🥷 [SILENT MODE] Lệnh gửi Browser: {} (Chip: {})",
                in_page_cmd.action.to_uppercase(),
                in_page_cmd.amount
            );
        }

        return Some(in_page_cmd);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{FEATURE_DIM, Street};
    use crate::live::protocol::{build_active_seats, process_game_update, robust_parse_card, HandActionRecord, PlayerTracker};
    use crate::live::PlayMode;
    use crate::neural_network::MLP;
    use serde_json::json;
    use std::sync::{Arc, RwLock};

    #[test]
    fn test_lounge_10max_sitting_out_excluded_and_advice_only() {
        let adv_mlp = MLP::new(&[FEATURE_DIM, 64, NUM_ACTIONS]);
        let policy = DeepCFRPolicy::new(Arc::new(RwLock::new(adv_mlp)), true);

        let mut state = HandState::new();
        state.hero_id = "p0".to_string();

        let mut players_map = serde_json::Map::new();
        let mut pgs_map = serde_json::Map::new();
        let mut seats_arr = Vec::new();
        for idx in 0..10 {
            let pid = format!("p{}", idx);
            seats_arr.push(json!([idx + 1, pid]));
            players_map.insert(
                pid.clone(),
                json!({ "stack": 3000, "name": format!("Player{}", idx), "seat": idx + 1 }),
            );
            pgs_map.insert(pid, json!(if idx == 9 { "away" } else { "inHand" }));
        }

        let payload = json!({
            "hI": "lounge_hand_1",
            "gT": [0, 0],
            "bigBlind": 20,
            "smallBlind": 10,
            "pot": 30,
            "cHB": 20,
            "dealerID": "p1",
            "seats": seats_arr,
            "players": players_map,
            "pGS": pgs_map,
            "pC": { "p0": { "cards": ["As", "Kd"] } },
            "pITT": "p0"
        });

        let cfg = PlayConfig::for_mode(PlayMode::Lounge);

        let active = {
            let mut advice_cfg = cfg.clone();
            advice_cfg.advice_only = true;
            let res_none = process_game_update(&mut state, &payload, &policy, &advice_cfg);
            assert!(res_none.is_none(), "advice_only=true must return None");
            assert!(state.has_evaluated_this_decision);
            build_active_seats(&state)
        };

        assert_eq!(active.len(), 9, "Sitting-out player p9 must be excluded from active seats");
        assert!(!active.contains(&"p9".to_string()));

        let ctx = build_environment(&state, state.hero_cards.unwrap(), PlayMode::Lounge).unwrap();
        assert!(ctx.env.state.is_cash_game, "Lounge mode must set is_cash_game = true");
        assert_eq!(ctx.env.state.num_players, 9);

        state.has_evaluated_this_decision = false;
        let act = process_game_update(&mut state, &payload, &policy, &cfg);
        assert!(act.is_some());
        let cmd = act.unwrap();
        assert!(["fold", "check", "call", "raise", "all_in"].contains(&cmd.action.as_str()));
    }

    #[test]
    fn test_lounge_subgame_solver_river_street_acted_and_overbet() {
        let adv_mlp = MLP::new(&[FEATURE_DIM, 64, NUM_ACTIONS]);
        let policy = DeepCFRPolicy::new(Arc::new(RwLock::new(adv_mlp)), true);

        let mut state = HandState::new();
        state.hand_id = "lounge_river_overbet".to_string();
        state.hero_id = "hero".to_string();
        state.street = Street::River;
        state.bb_size = 20;
        state.sb_size = 10;
        state.pn_total_pot = 200;
        state.highest_bet = 500;
        state.dealer_id = Some("hero".to_string());
        state.community_cards = vec![
            robust_parse_card("As").unwrap(),
            robust_parse_card("Kd").unwrap(),
            robust_parse_card("Qc").unwrap(),
            robust_parse_card("4s").unwrap(),
            robust_parse_card("5h").unwrap(),
        ];
        state.seats_order = vec!["hero".to_string(), "villain".to_string(), "folded_p".to_string()];

        let mut hero = PlayerTracker::default();
        hero.id = "hero".to_string();
        hero.name = "Hero".to_string();
        hero.seat_num = 1;
        hero.stack = 2000;
        hero.current_bet = 0;
        hero.total_contributed = 100;

        let mut villain = PlayerTracker::default();
        villain.id = "villain".to_string();
        villain.name = "Villain".to_string();
        villain.seat_num = 2;
        villain.stack = 2500;
        villain.current_bet = 500;
        villain.total_contributed = 600;

        let mut folded = PlayerTracker::default();
        folded.id = "folded_p".to_string();
        folded.name = "Folder".to_string();
        folded.seat_num = 3;
        folded.stack = 2000;
        folded.is_folded = true;
        folded.total_contributed = 20;

        state.players.insert("hero".to_string(), hero);
        state.players.insert("villain".to_string(), villain);
        state.players.insert("folded_p".to_string(), folded);
        state.action_history.push(HandActionRecord {
            player_id: "villain".to_string(),
            seat_num: 2,
            street: Street::River,
            action: Action::BetPot75,
            amount: 500,
        });

        let air_hole = [robust_parse_card("7c").unwrap(), robust_parse_card("2d").unwrap()];
        let ctx = build_environment(&state, air_hole, PlayMode::Lounge).unwrap();
        let villain_idx = ctx.active_seats.iter().position(|id| id == "villain").unwrap();
        assert!(ctx.env.street_has_acted[villain_idx]);
        assert!(!ctx.env.street_has_acted[ctx.hero_seat]);
        let mut sim_call = ctx.env.clone();
        assert!(sim_call.step(Action::CheckCall).unwrap());
        assert_eq!(sim_call.state.street, Street::Showdown);

        state.players.get_mut("villain").unwrap().stack = 500;
        state.players.get_mut("villain").unwrap().is_all_in = true;
        let mut cfg = PlayConfig::for_mode(PlayMode::Lounge);
        cfg.solver_config.iterations = 60;

        let fold_cmd = run_gto_evaluation(&state, air_hole, &policy, &cfg).unwrap();
        assert_eq!(fold_cmd.action, "fold", "72o facing 2.5x All-In overbet must fold");

        state.community_cards[0] = robust_parse_card("As").unwrap();
        state.community_cards[1] = robust_parse_card("Ks").unwrap();
        state.community_cards[2] = robust_parse_card("Qs").unwrap();
        let nuts_hole = [robust_parse_card("Ts").unwrap(), robust_parse_card("Jh").unwrap()];
        let nuts_cmd = run_gto_evaluation(&state, nuts_hole, &policy, &cfg).unwrap();
        assert_ne!(nuts_cmd.action, "fold", "Royal Flush must never fold");
    }

    #[test]
    fn test_tournament_overbet_broadway_straight_never_folds() {
        let adv_mlp = MLP::new(&[FEATURE_DIM, 64, NUM_ACTIONS]);
        let policy = DeepCFRPolicy::new(Arc::new(RwLock::new(adv_mlp)), true);

        let mut state = HandState::new();
        state.hand_id = "tour_overbet".to_string();
        state.hero_id = "hero1".to_string();
        state.street = Street::Turn;
        state.bb_size = 20;
        state.sb_size = 10;
        state.highest_bet = 500;
        state.pn_total_pot = 200;
        state.community_cards = vec![
            robust_parse_card("As").unwrap(),
            robust_parse_card("Kd").unwrap(),
            robust_parse_card("Qc").unwrap(),
            robust_parse_card("Jh").unwrap(),
        ];

        let mut hero = PlayerTracker::default();
        hero.id = "hero1".to_string();
        hero.name = "Hero".to_string();
        hero.stack = 2000;
        hero.current_bet = 0;
        hero.seat_num = 1;

        let mut opp = PlayerTracker::default();
        opp.id = "opp1".to_string();
        opp.name = "AggroWhale".to_string();
        opp.stack = 2000;
        opp.current_bet = 500;
        opp.seat_num = 2;

        state.players.insert(hero.id.clone(), hero);
        state.players.insert(opp.id.clone(), opp);
        state.seats_order = vec!["hero1".to_string(), "opp1".to_string()];

        let hole = [robust_parse_card("Tc").unwrap(), robust_parse_card("2s").unwrap()];
        let cfg = PlayConfig::for_mode(PlayMode::Tournament);

        let cmd = run_gto_evaluation(&state, hole, &policy, &cfg).unwrap();
        assert_ne!(cmd.action, "fold", "Broadway straight facing overbet must not fold");
    }

    #[test]
    fn test_tb_stale_bet_does_not_unfold_folded_player() {
        let adv_mlp = MLP::new(&[FEATURE_DIM, 64, NUM_ACTIONS]);
        let policy = DeepCFRPolicy::new(Arc::new(RwLock::new(adv_mlp)), true);
        let cfg = PlayConfig::for_mode(PlayMode::Lounge);

        let mut state = HandState::new();
        state.hero_id = "p0".to_string();

        // Packet 1: p1 bets 100
        let pkt1 = json!({
            "hI": "h_stale_fold",
            "gT": [0, 0],
            "bigBlind": 20,
            "smallBlind": 10,
            "pot": 150,
            "cHB": 100,
            "dealerID": "p0",
            "seats": [[1, "p0"], [2, "p1"], [3, "p2"]],
            "players": {
                "p0": { "stack": 2000, "name": "Hero", "seat": 1 },
                "p1": { "stack": 2000, "name": "P1", "seat": 2 },
                "p2": { "stack": 2000, "name": "P2", "seat": 3 }
            },
            "pGS": { "p0": "inHand", "p1": "inHand", "p2": "inHand" },
            "tB": { "p1": 100, "p2": 200 }
        });
        let _ = process_game_update(&mut state, &pkt1, &policy, &cfg);
        assert!(!state.players["p1"].is_folded);

        // Packet 2: p1 folds in pGS while stale tB still carries 100 (equal to prev_bet)
        let pkt2 = json!({
            "hI": "h_stale_fold",
            "pGS": { "p0": "inHand", "p1": "fold", "p2": "inHand" },
            "tB": { "p1": 100, "p2": 200 },
            "pC": { "p0": { "cards": ["Ah", "Kh"] } },
            "pITT": "p0"
        });
        let _ = process_game_update(&mut state, &pkt2, &policy, &PlayConfig { advice_only: true, ..cfg });
        assert!(
            state.players["p1"].is_folded,
            "Stale tB value must NEVER un-fold a player marked fold in pGS"
        );
    }

    #[test]
    fn test_3way_bet_and_call_marks_both_bettor_and_caller_acted() {
        let adv_mlp = MLP::new(&[FEATURE_DIM, 64, NUM_ACTIONS]);
        let policy = DeepCFRPolicy::new(Arc::new(RwLock::new(adv_mlp)), true);
        let mut cfg = PlayConfig::for_mode(PlayMode::Lounge);
        cfg.advice_only = true;

        let mut state = HandState::new();
        state.hero_id = "btn".to_string();

        let payload = json!({
            "hI": "h_3way_river",
            "gT": [0, 3],
            "bigBlind": 20,
            "smallBlind": 10,
            "pot": 300,
            "cHB": 200,
            "dealerID": "btn",
            "oTC": { "0": ["As", "7d", "2c"], "1": ["9h"], "2": ["4s"] },
            "seats": [[1, "btn"], [2, "sb"], [3, "bb"]],
            "players": {
                "btn": { "stack": 2000, "name": "Hero", "seat": 1 },
                "sb":  { "stack": 2000, "name": "SB", "seat": 2 },
                "bb":  { "stack": 2000, "name": "BB", "seat": 3 }
            },
            "pGS": { "btn": "inHand", "sb": "inHand", "bb": "inHand" },
            "tB": { "sb": 200, "bb": 200 },
            "pC": { "btn": { "cards": ["Ac", "Kc"] } },
            "pITT": "btn"
        });
        let _ = process_game_update(&mut state, &payload, &policy, &cfg);

        let ctx = build_environment(&state, state.hero_cards.unwrap(), PlayMode::Lounge).unwrap();
        let sb_idx = ctx.active_seats.iter().position(|id| id == "sb").unwrap();
        let bb_idx = ctx.active_seats.iter().position(|id| id == "bb").unwrap();
        assert!(ctx.env.street_has_acted[sb_idx]);
        assert!(ctx.env.street_has_acted[bb_idx]);
        assert!(!ctx.env.street_has_acted[ctx.hero_seat]);

        let mut sim = ctx.env.clone();
        assert!(
            sim.step(Action::CheckCall).unwrap(),
            "3-way River bet+call+hero_call must end hand at Showdown without giving caller another turn"
        );
        assert_eq!(sim.state.street, Street::Showdown);
    }
}

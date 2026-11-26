use crate::deep_cfr::DeepCFRPolicy;
use crate::environment::state::ActionRecord;
use crate::environment::{Action, Environment, MAX_PLAYERS, PlayerState, Street};
use crate::live::evaluator::run_gto_evaluation;
use crate::live::{InPageAction, PlayConfig, PlayMode};
use crate::poker_core::{Card, Rank, Suit};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct HandActionRecord {
    pub player_id: String,
    pub seat_num: usize,
    pub street: Street,
    pub action: Action,
    pub amount: u32,
}

#[derive(Clone, Debug)]
pub struct PlayerTracker {
    pub id: String,
    pub name: String,
    pub seat_num: usize,
    pub stack: u32,
    pub current_bet: u32,
    pub total_contributed: u32,
    pub is_folded: bool,
    pub is_all_in: bool,
    pub is_sitting_out: bool,
}

impl Default for PlayerTracker {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "Player".to_string(),
            seat_num: 99,
            stack: 1000,
            current_bet: 0,
            total_contributed: 0,
            is_folded: false,
            is_all_in: false,
            is_sitting_out: false,
        }
    }
}

#[derive(Debug)]
pub struct HandState {
    pub hand_id: String,
    pub hero_id: String,
    pub hero_cards: Option<[Card; 2]>,
    pub community_cards: Vec<Card>,
    pub street: Street,
    pub pn_total_pot: u32,
    pub highest_bet: u32,
    pub min_raise: u32,
    pub bb_size: u32,
    pub sb_size: u32,
    pub dealer_id: Option<String>,
    pub dealer_seat: Option<usize>,
    pub action_history: Vec<HandActionRecord>,
    pub players: HashMap<String, PlayerTracker>,
    pub seats_order: Vec<String>,
    pub has_evaluated_this_decision: bool,
    pub hero_mavtb: Option<u32>,
}

impl Default for HandState {
    fn default() -> Self {
        Self::new()
    }
}

impl HandState {
    pub fn new() -> Self {
        Self {
            hand_id: String::new(),
            hero_id: String::new(),
            hero_cards: None,
            community_cards: Vec::new(),
            street: Street::Preflop,
            pn_total_pot: 0,
            highest_bet: 0,
            min_raise: 20,
            bb_size: 20,
            sb_size: 10,
            dealer_id: None,
            dealer_seat: None,
            action_history: Vec::new(),
            players: HashMap::new(),
            seats_order: Vec::new(),
            has_evaluated_this_decision: false,
            hero_mavtb: None,
        }
    }

    pub fn start_new_hand(&mut self, new_h_id: &str) {
        println!("\n============================================================");
        println!("🎲 [VÁN MỚI] Hand ID: {}", new_h_id);
        println!("============================================================");
        self.hand_id = new_h_id.to_string();
        self.hero_cards = None;
        self.community_cards.clear();
        self.pn_total_pot = 0;
        self.highest_bet = 0;
        self.street = Street::Preflop;
        self.has_evaluated_this_decision = false;
        self.dealer_id = None;
        self.dealer_seat = None;
        self.action_history.clear();
        self.hero_mavtb = None;

        if !self.seats_order.is_empty() {
            self.players.retain(|id, _| self.seats_order.contains(id));
        }

        for p in self.players.values_mut() {
            p.current_bet = 0;
            p.total_contributed = 0;
            p.is_folded = false;
            p.is_all_in = false;
        }
    }
}

pub fn robust_parse_card(s: &str) -> Option<Card> {
    let s = s.trim();
    if s.len() < 2 || s.len() > 3 {
        return None;
    }
    let (r_char, s_char) = if s.len() == 3 && s.starts_with("10") {
        ('T', s.chars().nth(2)?)
    } else {
        let mut it = s.chars();
        (it.next()?, it.next()?)
    };

    let rank = match r_char {
        '2' => Rank::Two,
        '3' => Rank::Three,
        '4' => Rank::Four,
        '5' => Rank::Five,
        '6' => Rank::Six,
        '7' => Rank::Seven,
        '8' => Rank::Eight,
        '9' => Rank::Nine,
        'T' | 't' => Rank::Ten,
        'J' | 'j' => Rank::Jack,
        'Q' | 'q' => Rank::Queen,
        'K' | 'k' => Rank::King,
        'A' | 'a' => Rank::Ace,
        _ => return None,
    };
    let suit = match s_char {
        's' | 'S' => Suit::Spades,
        'h' | 'H' => Suit::Hearts,
        'd' | 'D' => Suit::Diamonds,
        'c' | 'C' => Suit::Clubs,
        _ => return None,
    };
    Some(Card::new(rank, suit))
}

pub fn parse_cards_array(val: &Value) -> Option<[Card; 2]> {
    let arr = val.as_array()?;
    let mut cards = Vec::new();
    for item in arr {
        if let Some(s) = item.as_str() {
            if let Some(card) = robust_parse_card(s) {
                cards.push(card);
            }
        } else if let Some(s) = item.get("value").and_then(|v| v.as_str()) {
            if let Some(card) = robust_parse_card(s) {
                cards.push(card);
            }
        }
    }
    if cards.len() == 2 {
        Some([cards[0], cards[1]])
    } else {
        None
    }
}

pub fn process_game_update(
    state: &mut HandState,
    payload: &Value,
    policy: &DeepCFRPolicy,
    cfg: &PlayConfig,
) -> Option<InPageAction> {
    if let Some(hi) = payload.get("hI").and_then(|v| v.as_str()) {
        if state.hand_id != hi {
            state.start_new_hand(hi);
        }
    }

    if let Some(gt_arr) = payload.get("gT").and_then(|v| v.as_array()) {
        if gt_arr.len() >= 2 {
            if let Some(st_num) = gt_arr[1].as_u64() {
                let new_street = match st_num {
                    0 => Street::Preflop,
                    1 => Street::Flop,
                    2 => Street::Turn,
                    3 => Street::River,
                    _ => Street::Showdown,
                };
                if new_street != state.street {
                    for p in state.players.values_mut() {
                        if p.is_all_in {
                            p.stack = 0;
                        } else {
                            p.stack = p.stack.saturating_sub(p.current_bet);
                        }
                        p.current_bet = 0;
                    }
                    state.highest_bet = 0;
                    state.street = new_street;
                    state.has_evaluated_this_decision = false;
                    state.hero_mavtb = None;
                }
            }
        }
    }

    if let Some(arr) = payload.get("seats").and_then(|v| v.as_array()) {
        let mut v: Vec<(usize, String)> = Vec::new();
        for s in arr {
            let (seat_num, pid_opt) = if let Some(a) = s.as_array() {
                let seat = a.first().and_then(|v| v.as_u64()).map(|n| n as usize);
                let pid = a
                    .get(1)
                    .and_then(|v| v.as_str())
                    .or_else(|| a.get(1).and_then(|v| v.get("id")).and_then(|v| v.as_str()))
                    .map(|s| s.to_string());
                (seat, pid)
            } else if let Some(obj) = s.as_object() {
                let seat = obj
                    .get("seat")
                    .or_else(|| obj.get("seatNum"))
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                let pid = obj
                    .get("id")
                    .or_else(|| obj.get("player_id"))
                    .or_else(|| obj.get("playerId"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                (seat, pid)
            } else {
                (None, None)
            };

            if let (Some(seat), Some(pid)) = (seat_num, pid_opt) {
                let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                    id: pid.clone(),
                    ..Default::default()
                });
                entry.seat_num = seat;
                v.push((seat, pid));
            }
        }
        if !v.is_empty() {
            v.sort_by_key(|(s, _)| *s);
            state.seats_order = v.into_iter().map(|(_, id)| id).collect();
        }
    } else if let Some(obj) = payload.get("seats").and_then(|v| v.as_object()) {
        let mut v: Vec<(usize, String)> = Vec::new();
        for (k, val) in obj {
            if let Ok(seat) = k.parse::<usize>() {
                if let Some(pid) = val.as_str().or_else(|| val.get("id").and_then(|v| v.as_str())) {
                    let entry = state.players.entry(pid.to_string()).or_insert_with(|| PlayerTracker {
                        id: pid.to_string(),
                        ..Default::default()
                    });
                    entry.seat_num = seat;
                    v.push((seat, pid.to_string()));
                }
            }
        }
        if !v.is_empty() {
            v.sort_by_key(|(s, _)| *s);
            state.seats_order = v.into_iter().map(|(_, id)| id).collect();
        }
    }

    if let Some(bb) = payload.get("bigBlind").and_then(|v| v.as_u64()) {
        state.bb_size = bb as u32;
    }
    if let Some(sb) = payload.get("smallBlind").and_then(|v| v.as_u64()) {
        state.sb_size = sb as u32;
    }

    if let Some(p) = payload.get("pot") {
        if let Some(num) = p.as_u64() {
            state.pn_total_pot = num as u32;
        } else if let Some(arr) = p.as_array() {
            state.pn_total_pot = arr.iter().filter_map(|v| v.as_u64()).sum::<u64>() as u32;
        }
    }

    let mut prev_highest = state.highest_bet;
    if let Some(chb) = payload.get("cHB").and_then(|v| v.as_u64()) {
        state.highest_bet = chb as u32;
    }

    if let Some(mr) = payload.get("mR").and_then(|v| v.as_u64()) {
        state.min_raise = (mr as u32).saturating_sub(state.highest_bet).max(state.bb_size);
    }

    if let Some(players_obj) = payload.get("players").and_then(|v| v.as_object()) {
        for (pid, pval) in players_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                id: pid.clone(),
                ..Default::default()
            });
            if let Some(s) = pval.get("stack").and_then(|v| v.as_u64()) {
                entry.stack = s as u32;
            }
            if let Some(n) = pval.get("name").and_then(|v| v.as_str()) {
                entry.name = n.to_string();
            }
            if let Some(seat) = pval.get("seat").or_else(|| pval.get("seatNum")).and_then(|v| v.as_u64()) {
                entry.seat_num = seat as usize;
            }
        }
    }

    for (idx, pid) in state.seats_order.iter().enumerate() {
        if let Some(p) = state.players.get_mut(pid) {
            if p.seat_num == 99 {
                p.seat_num = idx + 1;
            }
        }
    }

    if let Some(did) = payload.get("dealerID").and_then(|v| v.as_str()) {
        state.dealer_id = Some(did.to_string());
    }
    if let Some(ds) = payload
        .get("dealerSeat")
        .or_else(|| payload.get("dealer"))
        .and_then(|v| v.as_u64())
    {
        state.dealer_seat = Some(ds as usize);
    } else if let Some(ref did) = state.dealer_id {
        if let Some(p) = state.players.get(did) {
            if p.seat_num != 99 {
                state.dealer_seat = Some(p.seat_num);
            }
        }
    }

    if let Some(pgs_obj) = payload.get("pGS").and_then(|v| v.as_object()) {
        for (pid, sval) in pgs_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                id: pid.clone(),
                ..Default::default()
            });
            if let Some(st) = sval.as_str() {
                if st == "away" || st == "sittingOut" || st == "quit" || st == "stoodUp" || st == "quitted" {
                    entry.is_folded = true;
                    entry.is_sitting_out = true;
                } else if st == "fold" {
                    entry.is_folded = true;
                    entry.is_sitting_out = false;
                } else if st == "allIn" {
                    entry.is_folded = false;
                    entry.is_sitting_out = false;
                    if !entry.is_all_in {
                        entry.is_all_in = true;
                        if pid != &state.hero_id {
                            state.has_evaluated_this_decision = false;
                        }
                    }
                } else if st == "inHand" {
                    entry.is_folded = false;
                    entry.is_sitting_out = false;
                }
            }
        }
    }

    if let Some(tb_obj) = payload.get("tB").and_then(|v| v.as_object()) {
        for (pid, bval) in tb_obj {
            let entry = state.players.entry(pid.clone()).or_insert_with(|| PlayerTracker {
                id: pid.clone(),
                ..Default::default()
            });
            let prev_bet = entry.current_bet;
            if let Some(amt) = bval.as_u64() {
                let new_bet = amt as u32;
                if new_bet > prev_bet {
                    // Chỉ gỡ cờ fold khi đối thủ thực sự tăng mức cược ở vòng hiện tại
                    entry.is_folded = false;
                    entry.is_sitting_out = false;
                    let delta = new_bet - prev_bet;
                    entry.total_contributed += delta;

                    if new_bet > prev_highest && (state.street != Street::Preflop || new_bet > state.bb_size) {
                        if pid != &state.hero_id {
                            state.has_evaluated_this_decision = false;
                        }
                        let act_type = if entry.is_all_in {
                            Action::AllIn
                        } else if delta <= state.bb_size * 2 && state.street != Street::Preflop {
                            Action::BetSmall
                        } else if (delta as f32) < (state.pn_total_pot.max(state.bb_size * 2) as f32 * 0.50) {
                            Action::BetPot33
                        } else {
                            Action::BetPot75
                        };
                        state.action_history.push(HandActionRecord {
                            player_id: pid.clone(),
                            seat_num: entry.seat_num,
                            street: state.street,
                            action: act_type,
                            amount: delta,
                        });
                        prev_highest = new_bet;
                    } else if state.street != Street::Preflop
                        || new_bet > state.bb_size
                        || (prev_highest >= state.bb_size && new_bet == prev_highest)
                    {
                        state.action_history.push(HandActionRecord {
                            player_id: pid.clone(),
                            seat_num: entry.seat_num,
                            street: state.street,
                            action: Action::CheckCall,
                            amount: delta,
                        });
                    }
                }
                entry.current_bet = new_bet;
                if new_bet > state.highest_bet {
                    state.highest_bet = new_bet;
                }
            } else if bval.as_str() == Some("<D>") && !entry.is_all_in {
                entry.current_bet = 0;
            }
        }
    }

    if let Some(ihpi) = payload.get("iHPI").and_then(|v| v.as_array()) {
        let active_set: HashSet<String> =
            ihpi.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
        if !active_set.is_empty() {
            for (pid, p) in state.players.iter_mut() {
                let in_hand = active_set.contains(pid);
                p.is_folded = !in_hand;
                if in_hand {
                    p.is_sitting_out = false;
                }
            }
        }
    }

    if let Some(otc) = payload.get("oTC").and_then(|v| v.as_object()) {
        let mut sorted_keys: Vec<&String> = otc.keys().collect();
        sorted_keys.sort_by_key(|k| k.parse::<usize>().unwrap_or(99));
        let mut new_board = Vec::new();
        for k in sorted_keys {
            if let Some(arr) = otc.get(k).and_then(|v| v.as_array()) {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        if let Some(c) = robust_parse_card(s) {
                            new_board.push(c);
                        }
                    }
                }
            }
        }
        if !new_board.is_empty() && new_board != state.community_cards {
            println!("🎴 [BÀI CHUNG MỚI]: {:?}", new_board);
            state.community_cards = new_board;
            state.has_evaluated_this_decision = false;
        }
    }

    if let Some(pc_obj) = payload.get("pC").and_then(|v| v.as_object()) {
        for (pid, cdata) in pc_obj {
            let is_showdown = cdata
                .get("cards")
                .and_then(|c| c.as_array())
                .is_some_and(|arr| arr.iter().any(|item| item.get("showing") == Some(&Value::Bool(true))));
            if is_showdown {
                continue;
            }

            let cards_val = cdata.get("cards").or_else(|| cdata.get("c"));
            if let Some(c_val) = cards_val {
                if let Some(parsed) = parse_cards_array(c_val) {
                    if state.hero_id.is_empty() || pid == &state.hero_id {
                        state.hero_id = pid.clone();
                        if state.hero_cards != Some(parsed) {
                            state.hero_cards = Some(parsed);
                            println!("🎴 [BÀI TẨY CỦA BẠN]: [ {} {} ] (Hero ID: {})", parsed[0], parsed[1], pid);
                        }
                    }
                }
            }
        }
    }

    let current_actor = payload.get("pITT").or_else(|| payload.get("cPI")).and_then(|v| v.as_str());

    if let Some(actor) = current_actor {
        if let Some(mavtb) = payload.get("mAVTB").and_then(|v| v.as_u64()) {
            let mavtb_u32 = mavtb as u32;
            if actor == state.hero_id {
                state.hero_mavtb = Some(mavtb_u32);
                if let Some(p) = state.players.get_mut(actor) {
                    p.stack = mavtb_u32;
                }
            }
        }

        if !state.hero_id.is_empty() && actor == state.hero_id {
            if let Some(hero_p) = state.players.get_mut(&state.hero_id) {
                hero_p.is_folded = false;
                hero_p.is_sitting_out = false;
            }
            if !state.has_evaluated_this_decision {
                if let Some(hole) = state.hero_cards {
                    let action = run_gto_evaluation(state, hole, policy, cfg);
                    if action.is_some() || cfg.advice_only {
                        state.has_evaluated_this_decision = true;
                    }
                    return action;
                } else {
                    println!("⏳ Đến lượt bạn nhưng ván này chưa nhận diện được bài tẩy!");
                }
            }
        } else {
            state.has_evaluated_this_decision = false;
        }
    }

    None
}

pub fn build_active_seats(state: &HandState) -> Vec<String> {
    let mut sorted_players: Vec<&PlayerTracker> = state.players.values().collect();
    sorted_players.sort_by_key(|p| p.seat_num);

    let mut active_seats: Vec<String> = sorted_players
        .iter()
        .filter(|p| {
            if p.id == state.hero_id {
                return true;
            }
            if p.is_sitting_out {
                return false;
            }
            let stack_behind = if p.is_all_in {
                0
            } else {
                p.stack.saturating_sub(p.current_bet)
            };
            stack_behind > 0 || p.current_bet > 0 || p.is_all_in
        })
        .map(|p| p.id.clone())
        .collect();

    if active_seats.len() < 2 {
        for p in &sorted_players {
            if !p.is_sitting_out && !active_seats.contains(&p.id) {
                active_seats.push(p.id.clone());
                if active_seats.len() >= 2 {
                    break;
                }
            }
        }
    }
    if active_seats.len() < 2 {
        for p in &sorted_players {
            if !active_seats.contains(&p.id) {
                active_seats.push(p.id.clone());
                if active_seats.len() >= 2 {
                    break;
                }
            }
        }
    }
    active_seats
}

pub struct EvalContext {
    pub env: Environment,
    pub active_seats: Vec<String>,
    pub hero_seat: usize,
    pub actual_highest_bet: u32,
    pub true_table_pot: u32,
    pub safe_min_raise: u32,
    pub to_call: u32,
    pub max_total_bet: u32,
    pub can_raise: bool,
    pub in_hand_count: usize,
}

pub fn build_environment(state: &HandState, hole: [Card; 2], mode: PlayMode) -> Option<EvalContext> {
    let mut active_seats = build_active_seats(state);
    if active_seats.is_empty() {
        return None;
    }

    let num_players = active_seats.len().clamp(2, MAX_PLAYERS);
    if let Some(h_pos) = active_seats.iter().position(|id| id == &state.hero_id) {
        if h_pos >= num_players {
            active_seats.swap(0, h_pos);
        }
    }
    active_seats.truncate(num_players);
    let hero_seat = active_seats.iter().position(|id| id == &state.hero_id).unwrap_or(0);

    let mut actual_highest_bet = state.highest_bet;
    let mut sum_current_bets = 0u32;
    let mut initial_stacks = Vec::with_capacity(num_players);
    let mut env_players = Vec::with_capacity(num_players);

    for (seat_idx, pid) in active_seats.iter().enumerate() {
        let p_track = state.players.get(pid).cloned().unwrap_or_default();
        if p_track.current_bet > actual_highest_bet {
            actual_highest_bet = p_track.current_bet;
        }
        let true_stack_behind = if p_track.is_all_in {
            0
        } else {
            p_track.stack.saturating_sub(p_track.current_bet)
        };
        let starting_stack = true_stack_behind + p_track.current_bet;
        initial_stacks.push(starting_stack.max(state.bb_size));

        let mut ps = PlayerState::new(seat_idx, true_stack_behind, seat_idx != hero_seat);
        ps.current_bet = p_track.current_bet;
        ps.total_contributed = p_track.total_contributed.max(p_track.current_bet);
        ps.is_folded = if seat_idx == hero_seat { false } else { p_track.is_folded };
        ps.is_all_in = p_track.is_all_in;
        if seat_idx == hero_seat {
            ps.hole_cards = [Some(hole[0]), Some(hole[1])];
        }

        sum_current_bets += p_track.current_bet;
        env_players.push(ps);
    }

    while initial_stacks.len() < num_players {
        let opp_stack = actual_highest_bet.max(state.bb_size * 50);
        initial_stacks.push(opp_stack);
        let dummy_seat = env_players.len();
        let mut ps = PlayerState::new(dummy_seat, opp_stack, true);
        ps.current_bet = actual_highest_bet;
        ps.total_contributed = actual_highest_bet;
        env_players.push(ps);
    }
    initial_stacks.truncate(num_players);
    env_players.truncate(num_players);

    let in_hand_count = env_players.iter().filter(|pl| !pl.is_folded).count();

    let button_idx = if let Some(ref did) = state.dealer_id {
        active_seats.iter().position(|id| id == did).unwrap_or(0)
    } else if let Some(d_seat) = state.dealer_seat {
        active_seats
            .iter()
            .position(|id| state.players.get(id).map(|p| p.seat_num) == Some(d_seat))
            .unwrap_or(0)
    } else {
        0
    };

    let mut env = Environment::new(num_players, &initial_stacks, state.bb_size, state.sb_size, button_idx);
    env.state.is_cash_game = mode.is_cash_game();

    let true_table_pot = (state.pn_total_pot + sum_current_bets).max(state.bb_size + state.sb_size);
    let safe_min_raise = state.min_raise.max(state.bb_size);

    env.state.pot = true_table_pot;
    env.state.current_highest_bet = actual_highest_bet;
    env.state.min_raise = safe_min_raise;
    env.state.community_cards = state.community_cards.clone();
    env.state.street = state.street;
    env.state.current_player_idx = hero_seat;
    env.state.button_idx = button_idx;
    env.state.players = env_players;

    env.state.action_history = state
        .action_history
        .iter()
        .filter_map(|rec| {
            let idx = active_seats.iter().position(|id| id == &rec.player_id)?;
            Some(ActionRecord {
                player_idx: idx,
                street: rec.street,
                action: rec.action,
                amount: rec.amount,
            })
        })
        .collect();

    // Khôi phục chính xác street_has_acted cho cả người bet lẫn người call trước Hero
    env.street_has_acted = vec![false; num_players];
    let mut saw_current_street_action = false;
    for rec in &env.state.action_history {
        if rec.street == state.street && rec.player_idx < num_players {
            saw_current_street_action = true;
            match rec.action {
                Action::BetSmall | Action::BetPot33 | Action::BetPot75 | Action::AllIn => {
                    env.street_has_acted.fill(false);
                    env.street_has_acted[rec.player_idx] = true;
                }
                Action::CheckCall | Action::Fold => {
                    env.street_has_acted[rec.player_idx] = true;
                }
            }
        }
    }
    if !saw_current_street_action && state.street != Street::Preflop && actual_highest_bet == 0 {
        let mut cursor = (button_idx + 1) % num_players;
        for _ in 0..num_players {
            if cursor == hero_seat {
                break;
            }
            if env.state.players[cursor].is_active() {
                env.street_has_acted[cursor] = true;
            }
            cursor = (cursor + 1) % num_players;
        }
    }
    if actual_highest_bet > 0 && !(state.street == Street::Preflop && actual_highest_bet == state.bb_size) {
        for (i, pl) in env.state.players.iter().enumerate() {
            if i != hero_seat && pl.current_bet == actual_highest_bet {
                env.street_has_acted[i] = true;
            }
        }
    }
    env.street_has_acted[hero_seat] = false;

    let hero_bet = state.players.get(&state.hero_id).map(|p| p.current_bet).unwrap_or(0);
    let hero_stack_behind = env.state.players[hero_seat].stack;
    let to_call = actual_highest_bet.saturating_sub(hero_bet);
    let max_total_bet = state.hero_mavtb.unwrap_or(hero_stack_behind + hero_bet);

    let has_opp_with_chips = active_seats.iter().any(|pid| {
        if pid == &state.hero_id {
            return false;
        }
        if let Some(p) = state.players.get(pid) {
            let opp_stack_behind = if p.is_all_in {
                0
            } else {
                p.stack.saturating_sub(p.current_bet)
            };
            !p.is_folded && !p.is_all_in && !p.is_sitting_out && opp_stack_behind > 0
        } else {
            false
        }
    });
    let has_chips_to_raise = hero_stack_behind > to_call;
    let mavtb_allows_raise = state.hero_mavtb.is_none_or(|m| m > actual_highest_bet);
    let can_raise = has_opp_with_chips && has_chips_to_raise && mavtb_allows_raise;

    Some(EvalContext {
        env,
        active_seats,
        hero_seat,
        actual_highest_bet,
        true_table_pot,
        safe_min_raise,
        to_call,
        max_total_bet,
        can_raise,
        in_hand_count,
    })
}

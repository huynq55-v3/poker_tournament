//! Bước 1 của bot RL: mã hóa trạng thái ván heads-up thành chuỗi token cho transformer.
//!
//! Chuỗi token (theo thứ tự):
//!   `[STATE] [HOLE] [HOLE] [BOARD]×0..5 [ACTION]×0..MAX_ACTIONS [PAD]…`
//!
//! Mỗi token gồm các trường rời rạc (sẽ được embed rồi cộng lại) và `CONT_DIM` đặc trưng liên tục:
//! - `STATE`: stack của mình / đối thủ, pot, số chip cần call, mình có phải button, street, SPR.
//! - `HOLE` / `BOARD`: rank + suit của lá bài (suit đã chuẩn hóa, xem dưới).
//! - `ACTION`: ai đánh (mình / đối thủ), street, loại hành động, số chip bỏ vào so với pot lúc đó.
//!
//! Chuẩn hóa suit: hoán vị suit không đổi bản chất ván bài, nên suit được đánh số lại theo thứ tự
//! xuất hiện đầu tiên (bài tẩy đã sắp xếp → flop đã sắp xếp → turn → river). Nhờ vậy `A♥K♥` và
//! `A♠K♠` cho cùng một mã hóa. Mapping chỉ phụ thuộc vào các lá đã lộ nên ổn định suốt ván.
//!
//! Chỉ dùng thông tin người chơi `player` được phép thấy: không bao giờ đọc bài tẩy đối thủ.
//!
//! Đơn vị chip: chia cho `100 × BB` (stack khởi điểm heads-up), nên stack đầy = 1.0.

use crate::environment::{GameState, Street};
use crate::poker_core::Card;

/// Số hành động lịch sử tối đa được giữ (giữ các hành động gần nhất nếu vượt).
pub const MAX_ACTIONS: usize = 40;
/// 1 STATE + 2 HOLE + 5 BOARD + MAX_ACTIONS.
pub const MAX_TOKENS: usize = 1 + 2 + 5 + MAX_ACTIONS;
/// Số đặc trưng liên tục mỗi token.
pub const CONT_DIM: usize = 8;

/// Loại token.
pub const KIND_PAD: u8 = 0;
pub const KIND_STATE: u8 = 1;
pub const KIND_HOLE: u8 = 2;
pub const KIND_BOARD: u8 = 3;
pub const KIND_ACTION: u8 = 4;

/// Kích thước từ vựng của từng trường rời rạc (giá trị cuối cùng = "không áp dụng").
pub const KIND_VOCAB: usize = 5;
pub const RANK_VOCAB: usize = 14; // 0..12 + 13 (none)
pub const SUIT_VOCAB: usize = 5; // 0..3 + 4 (none)
pub const ACTOR_VOCAB: usize = 3; // 0 = mình, 1 = đối thủ, 2 (none)
pub const STREET_VOCAB: usize = 5; // Preflop..River + 4 (none)
pub const ACTION_VOCAB: usize = 7; // 6 hành động + 6 (none)

const RANK_NONE: u8 = 13;
const SUIT_NONE: u8 = 4;
const ACTOR_NONE: u8 = 2;
const STREET_NONE: u8 = 4;
const ACTION_NONE: u8 = 6;

/// Số trường rời rạc mỗi token: kind, rank, suit, actor, street, action.
pub const DISCRETE_FIELDS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Token {
    pub kind: u8,
    pub rank: u8,
    pub suit: u8,
    pub actor: u8,
    pub street: u8,
    pub action: u8,
    pub cont: [f32; CONT_DIM],
}

impl Token {
    const PAD: Token = Token {
        kind: KIND_PAD,
        rank: RANK_NONE,
        suit: SUIT_NONE,
        actor: ACTOR_NONE,
        street: STREET_NONE,
        action: ACTION_NONE,
        cont: [0.0; CONT_DIM],
    };

    fn discrete(&self) -> [u8; DISCRETE_FIELDS] {
        [self.kind, self.rank, self.suit, self.actor, self.street, self.action]
    }
}

/// Trạng thái đã mã hóa, độ dài cố định `MAX_TOKENS` (phần thừa là PAD).
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedState {
    pub tokens: [Token; MAX_TOKENS],
    /// Số token thật (không tính PAD).
    pub len: usize,
    /// Số hành động lịch sử bị cắt bỏ vì vượt `MAX_ACTIONS` (thường = 0).
    pub truncated_actions: usize,
}

impl EncodedState {
    /// Ghi các trường rời rạc vào `out` (độ dài `MAX_TOKENS * DISCRETE_FIELDS`), dạng i64 cho embedding.
    pub fn write_discrete(&self, out: &mut [i64]) {
        debug_assert_eq!(out.len(), MAX_TOKENS * DISCRETE_FIELDS);
        for (t, chunk) in self.tokens.iter().zip(out.chunks_exact_mut(DISCRETE_FIELDS)) {
            for (o, v) in chunk.iter_mut().zip(t.discrete()) {
                *o = v as i64;
            }
        }
    }

    /// Ghi đặc trưng liên tục vào `out` (độ dài `MAX_TOKENS * CONT_DIM`).
    pub fn write_cont(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), MAX_TOKENS * CONT_DIM);
        for (t, chunk) in self.tokens.iter().zip(out.chunks_exact_mut(CONT_DIM)) {
            chunk.copy_from_slice(&t.cont);
        }
    }

    /// Mặt nạ PAD cho attention, theo quy ước của `burn`: `true` = token PAD (bị che).
    pub fn write_pad_mask(&self, out: &mut [bool]) {
        debug_assert_eq!(out.len(), MAX_TOKENS);
        for (i, o) in out.iter_mut().enumerate() {
            *o = i >= self.len;
        }
    }
}

fn street_id(s: Street) -> u8 {
    match s {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
        Street::Showdown => 3,
    }
}

/// Nén tỉ lệ không chặn trên (vd. all-in 66× pot) về [0, 1): x / (1 + x). 1× pot → 0.5.
fn squash(x: f32) -> f32 {
    x / (1.0 + x)
}

/// Đánh số lại suit theo thứ tự xuất hiện đầu tiên.
struct SuitCanon {
    map: [u8; 4],
    next: u8,
}

impl SuitCanon {
    fn new() -> Self {
        SuitCanon { map: [SUIT_NONE; 4], next: 0 }
    }
    fn get(&mut self, c: Card) -> u8 {
        let s = c.suit_val() as usize;
        if self.map[s] == SUIT_NONE {
            self.map[s] = self.next;
            self.next += 1;
        }
        self.map[s]
    }
}

/// Sắp xếp rank giảm dần (rồi theo suit) để thứ tự lá trong nhóm không ảnh hưởng mã hóa.
fn sort_desc(cards: &mut [Card]) {
    cards.sort_by(|a, b| b.rank_val().cmp(&a.rank_val()).then(a.suit_val().cmp(&b.suit_val())));
}

/// Mã hóa trạng thái từ góc nhìn `player` (heads-up).
pub fn encode(state: &GameState, player: usize) -> EncodedState {
    assert_eq!(state.num_players, 2, "encoding RL hiện chỉ hỗ trợ heads-up");
    let opp = 1 - player;
    let unit = (state.bb_size.max(1) as f32) * 100.0;

    let mut tokens = [Token::PAD; MAX_TOKENS];
    let mut n = 0usize;

    // --- STATE ---
    let me = &state.players[player];
    let op = &state.players[opp];
    let to_call = state.current_highest_bet.saturating_sub(me.current_bet);
    let eff_stack = me.stack.min(op.stack + op.current_bet.saturating_sub(me.current_bet));
    let pot = state.pot.max(1) as f32;
    tokens[n] = Token {
        kind: KIND_STATE,
        street: street_id(state.street),
        cont: [
            me.stack as f32 / unit,
            op.stack as f32 / unit,
            state.pot as f32 / unit,
            to_call as f32 / unit,
            to_call as f32 / (pot + to_call as f32),
            if state.button_idx == player { 1.0 } else { 0.0 },
            (eff_stack as f32 / pot).min(20.0) / 20.0,
            state.min_raise as f32 / unit,
        ],
        ..Token::PAD
    };
    n += 1;

    // --- Bài: chuẩn hóa suit ---
    let mut canon = SuitCanon::new();
    let mut hole: Vec<Card> = me.hole_cards.iter().flatten().copied().collect();
    sort_desc(&mut hole);
    let board = &state.community_cards;
    let mut flop: Vec<Card> = board.iter().take(3).copied().collect();
    sort_desc(&mut flop);
    let ordered_board: Vec<Card> = flop.into_iter().chain(board.iter().skip(3).copied()).collect();

    for &c in &hole {
        tokens[n] = Token { kind: KIND_HOLE, rank: c.rank_val(), suit: canon.get(c), ..Token::PAD };
        n += 1;
    }
    for (i, &c) in ordered_board.iter().take(5).enumerate() {
        let street = if i < 3 { 1 } else { i as u8 - 1 };
        tokens[n] = Token { kind: KIND_BOARD, rank: c.rank_val(), suit: canon.get(c), street, ..Token::PAD };
        n += 1;
    }

    // --- ACTION: replay để biết pot và số chip cần call trước mỗi hành động ---
    let hist = &state.action_history;
    // Blind đã post = tổng đóng góp − tổng các hành động trong lịch sử.
    let mut spent_in_hist = [0u32; 2];
    for r in hist {
        if r.player_idx < 2 {
            spent_in_hist[r.player_idx] += r.amount;
        }
    }
    let mut bets = [0u32; 2];
    for i in 0..2 {
        bets[i] = state.players[i].total_contributed.saturating_sub(spent_in_hist[i]);
    }
    let mut pot_run: u32 = bets[0] + bets[1];
    let mut cur_street = Street::Preflop;
    let mut action_tokens: Vec<Token> = Vec::with_capacity(hist.len());
    for r in hist {
        if r.player_idx >= 2 {
            continue;
        }
        if r.street != cur_street {
            cur_street = r.street;
            bets = [0, 0];
        }
        let highest = bets[0].max(bets[1]);
        let to_call_before = highest.saturating_sub(bets[r.player_idx]);
        let pot_before = pot_run.max(1) as f32;
        bets[r.player_idx] += r.amount;
        pot_run += r.amount;
        let raise_by = bets[r.player_idx].saturating_sub(highest);
        action_tokens.push(Token {
            kind: KIND_ACTION,
            actor: if r.player_idx == player { 0 } else { 1 },
            street: street_id(r.street),
            action: r.action.to_index() as u8,
            cont: [
                squash(r.amount as f32 / pot_before),
                r.amount as f32 / unit,
                pot_before / unit,
                to_call_before as f32 / unit,
                squash(raise_by as f32 / (pot_before + to_call_before as f32)),
                0.0,
                0.0,
                0.0,
            ],
            ..Token::PAD
        });
    }
    let truncated_actions = action_tokens.len().saturating_sub(MAX_ACTIONS);
    for t in action_tokens.into_iter().skip(truncated_actions) {
        tokens[n] = t;
        n += 1;
    }

    EncodedState { tokens, len: n, truncated_actions }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{Action, Environment};
    use crate::poker_core::{Rank, Suit};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const BB: u32 = 100;

    fn hu_env(seed: u64) -> Environment {
        let mut env = Environment::new(2, &[100 * BB, 100 * BB], BB, BB / 2, 0);
        let mut rng = StdRng::seed_from_u64(seed);
        env.reset_hand(&mut rng, 0);
        env
    }

    fn c(r: Rank, s: Suit) -> Card {
        Card::new(r, s)
    }

    #[test]
    fn preflop_state_token_and_layout() {
        let env = hu_env(1);
        let p = env.state.current_player_idx;
        assert_eq!(p, env.state.button_idx, "HU: button = SB hành động trước preflop");
        let e = encode(&env.state, p);
        assert_eq!(e.len, 3, "STATE + 2 HOLE, chưa có board/hành động");
        assert_eq!(e.tokens[0].kind, KIND_STATE);
        assert_eq!(e.tokens[1].kind, KIND_HOLE);
        assert_eq!(e.tokens[3].kind, KIND_PAD);
        let s = e.tokens[0].cont;
        assert!((s[0] - 0.995).abs() < 1e-6, "stack SB = 99.5bb");
        assert!((s[1] - 0.99).abs() < 1e-6, "stack BB = 99bb");
        assert!((s[2] - 0.015).abs() < 1e-6, "pot = 1.5bb");
        assert!((s[3] - 0.005).abs() < 1e-6, "cần call 0.5bb");
        assert_eq!(s[5], 1.0, "SB là button");
    }

    #[test]
    fn suit_permutation_gives_identical_encoding() {
        let mut a = hu_env(2);
        let mut b = a.clone();
        let p = a.state.current_player_idx;
        a.state.players[p].hole_cards = [Some(c(Rank::Ace, Suit::Hearts)), Some(c(Rank::King, Suit::Hearts))];
        b.state.players[p].hole_cards = [Some(c(Rank::King, Suit::Spades)), Some(c(Rank::Ace, Suit::Spades))];
        a.state.community_cards =
            vec![c(Rank::Two, Suit::Clubs), c(Rank::Nine, Suit::Hearts), c(Rank::Ten, Suit::Diamonds)];
        b.state.community_cards =
            vec![c(Rank::Ten, Suit::Hearts), c(Rank::Two, Suit::Diamonds), c(Rank::Nine, Suit::Spades)];
        assert_eq!(encode(&a.state, p), encode(&b.state, p));

        // Đổi màu khác bản chất (không còn đồng chất) thì mã hóa phải khác.
        b.state.players[p].hole_cards = [Some(c(Rank::King, Suit::Clubs)), Some(c(Rank::Ace, Suit::Spades))];
        assert_ne!(encode(&a.state, p), encode(&b.state, p));
    }

    #[test]
    fn opponent_hole_cards_never_leak() {
        let a = hu_env(3);
        let mut b = a.clone();
        let p = a.state.current_player_idx;
        b.state.players[1 - p].hole_cards = [Some(c(Rank::Two, Suit::Clubs)), Some(c(Rank::Seven, Suit::Diamonds))];
        assert_eq!(encode(&a.state, p), encode(&b.state, p));
    }

    #[test]
    fn action_tokens_track_pot_and_to_call() {
        let mut env = hu_env(4);
        let sb = env.state.current_player_idx;
        env.step(Action::BetPot75).unwrap(); // SB open
        let bb = env.state.current_player_idx;
        assert_ne!(sb, bb);
        let e = encode(&env.state, bb);
        let act = e.tokens[e.len - 1];
        assert_eq!(act.kind, KIND_ACTION);
        assert_eq!(act.actor, 1, "từ góc nhìn BB, SB là đối thủ");
        assert_eq!(act.action, Action::BetPot75.to_index() as u8);
        assert!((act.cont[2] - 0.015).abs() < 1e-6, "pot trước hành động = 1.5bb");
        assert!((act.cont[3] - 0.005).abs() < 1e-6, "SB cần call 0.5bb trước khi raise");
        let committed = act.cont[1] * 100.0; // theo bb
        assert!(committed > 0.5, "SB bỏ vào nhiều hơn phần call");

        env.step(Action::CheckCall).unwrap(); // BB call → flop
        assert_eq!(env.state.street, Street::Flop);
        let p = env.state.current_player_idx;
        let e = encode(&env.state, p);
        assert_eq!(e.len, 1 + 2 + 3 + 2);
        let call = e.tokens[e.len - 1];
        assert_eq!(call.action, Action::CheckCall.to_index() as u8);
        assert!(call.cont[4].abs() < 1e-6, "call không phải raise");
        // pot ở STATE khớp tổng chip đã vào
        let pot_from_tokens = call.cont[2] + call.cont[1];
        assert!((pot_from_tokens - e.tokens[0].cont[2]).abs() < 1e-5);
    }

    #[test]
    fn random_rollouts_encode_finite_bounded_and_flatten() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut disc = vec![0i64; MAX_TOKENS * DISCRETE_FIELDS];
        let mut cont = vec![0f32; MAX_TOKENS * CONT_DIM];
        let mut mask = vec![false; MAX_TOKENS];
        let mut decisions = 0;
        for hand in 0..300u64 {
            let mut env = hu_env(100 + hand);
            loop {
                let p = env.state.current_player_idx;
                let e = encode(&env.state, p);
                assert!(e.len <= MAX_TOKENS);
                assert_eq!(e.truncated_actions, 0, "ván ngẫu nhiên không nên vượt {MAX_ACTIONS} hành động");
                for t in &e.tokens[..e.len] {
                    assert!(t.cont.iter().all(|v| v.is_finite() && (-0.01..=2.01).contains(v)), "{:?}", t);
                    assert!((t.kind as usize) < KIND_VOCAB && (t.rank as usize) < RANK_VOCAB);
                    assert!((t.suit as usize) < SUIT_VOCAB && (t.actor as usize) < ACTOR_VOCAB);
                    assert!((t.street as usize) < STREET_VOCAB && (t.action as usize) < ACTION_VOCAB);
                }
                e.write_discrete(&mut disc);
                e.write_cont(&mut cont);
                e.write_pad_mask(&mut mask);
                assert_eq!(mask.iter().filter(|&&m| !m).count(), e.len);
                decisions += 1;

                let valid = env.get_action_mask().valid_actions();
                let a = valid[rng.gen_range(0..valid.len())];
                if env.step(a).unwrap() {
                    break;
                }
            }
        }
        assert!(decisions > 600);
    }
}

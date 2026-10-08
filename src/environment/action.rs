use serde::{Deserialize, Serialize};

pub const NUM_ACTIONS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Action {
    Fold = 0,
    CheckCall = 1,
    BetPot33 = 2,
    BetPot66 = 3,
    BetPot100 = 4,
    AllIn = 5,
}

impl Action {
    pub const ALL: [Action; NUM_ACTIONS] = [
        Action::Fold,
        Action::CheckCall,
        Action::BetPot33,
        Action::BetPot66,
        Action::BetPot100,
        Action::AllIn,
    ];

    #[inline(always)]
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Action::Fold),
            1 => Some(Action::CheckCall),
            2 => Some(Action::BetPot33),
            3 => Some(Action::BetPot66),
            4 => Some(Action::BetPot100),
            5 => Some(Action::AllIn),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn to_index(self) -> usize {
        self as usize
    }

    pub fn name(&self) -> &'static str {
        match self {
            Action::Fold => "Fold",
            Action::CheckCall => "Check/Call",
            Action::BetPot33 => "Bet 33% Pot",
            Action::BetPot66 => "Bet 66% Pot",
            Action::BetPot100 => "Bet 100% Pot",
            Action::AllIn => "All-In",
        }
    }
}

/// A compact bitmask representing valid actions
/// Bit 0: Fold, Bit 1: CheckCall, Bit 2: Bet33, Bit 3: Bet66, Bit 4: Bet100, Bit 5: AllIn
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionMask(pub u8);

impl ActionMask {
    pub const EMPTY: Self = Self(0);

    #[inline(always)]
    pub fn enable(&mut self, action: Action) {
        self.0 |= 1 << action.to_index();
    }

    #[inline(always)]
    pub fn is_valid(&self, action: Action) -> bool {
        (self.0 & (1 << action.to_index())) != 0
    }

    #[inline(always)]
    pub fn is_index_valid(&self, idx: usize) -> bool {
        if idx >= NUM_ACTIONS {
            false
        } else {
            (self.0 & (1 << idx)) != 0
        }
    }

    pub fn to_bool_array(&self) -> [bool; NUM_ACTIONS] {
        let mut arr = [false; NUM_ACTIONS];
        for i in 0..NUM_ACTIONS {
            arr[i] = (self.0 & (1 << i)) != 0;
        }
        arr
    }

    pub fn valid_actions(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(NUM_ACTIONS);
        for &act in &Action::ALL {
            if self.is_valid(act) {
                actions.push(act);
            }
        }
        actions
    }
}

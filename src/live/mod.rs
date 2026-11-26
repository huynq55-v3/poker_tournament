pub mod evaluator;
pub mod os_input;
pub mod protocol;

use crate::deep_cfr::SubgameSolverConfig;
pub use evaluator::run_gto_evaluation;
pub use os_input::{execute_bot_action, set_target_game};
pub use protocol::{build_active_seats, build_environment, parse_cards_array, process_game_update, robust_parse_card, EvalContext, HandActionRecord, HandState, PlayerTracker};

pub const DEFAULT_TOURNAMENT_ADV_PATH: &str = "deep_cfr_model.json";
pub const DEFAULT_TOURNAMENT_STRAT_PATH: &str = "deep_cfr_strategy.json";
pub const DEFAULT_LOUNGE_ADV_PATH: &str = "deep_cfr_lounge_model.json";
pub const DEFAULT_LOUNGE_STRAT_PATH: &str = "deep_cfr_lounge_strategy.json";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlayMode {
    Tournament,
    Lounge,
}

impl PlayMode {
    #[inline]
    pub fn is_cash_game(self) -> bool {
        matches!(self, PlayMode::Lounge)
    }

    pub fn default_model_paths(self) -> (&'static str, &'static str) {
        match self {
            PlayMode::Tournament => (DEFAULT_TOURNAMENT_ADV_PATH, DEFAULT_TOURNAMENT_STRAT_PATH),
            PlayMode::Lounge => (DEFAULT_LOUNGE_ADV_PATH, DEFAULT_LOUNGE_STRAT_PATH),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PlayMode::Tournament => "TOURNAMENT (ICM)",
            PlayMode::Lounge => "LOUNGE CASH (CHIP-EV / 12-300+ BB)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecisionMode {
    PureMaxEv,     // Mặc định: EV cao nhất
    GuardedMixing, // --guarded: Lọc nhánh tự sát / trộn an toàn
    PureGtoRoll,   // --roll: Tung xúc xắc theo % GTO
}

#[derive(Clone, Debug)]
pub struct PlayConfig {
    pub mode: PlayMode,
    pub decision_mode: DecisionMode,
    pub use_os_enigo: bool,
    pub advice_only: bool,
    pub delay_min_ms: u64,
    pub delay_max_ms: u64,
    pub enable_solver: bool,
    pub force_search: bool,
    pub solver_config: SubgameSolverConfig,
}

impl PlayConfig {
    pub fn for_mode(mode: PlayMode) -> Self {
        let solver_config = match mode {
            PlayMode::Tournament => SubgameSolverConfig::default(),
            PlayMode::Lounge => SubgameSolverConfig::lounge_cash_default(),
        };
        Self {
            mode,
            decision_mode: DecisionMode::PureMaxEv,
            use_os_enigo: false,
            advice_only: false,
            delay_min_ms: 1500,
            delay_max_ms: 3000,
            enable_solver: true,
            force_search: false,
            solver_config,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct InPageAction {
    pub action: String, // "fold", "check", "call", "all_in", "raise"
    pub amount: u32,
    pub to_call: u32,
    #[serde(default)]
    pub delay_ms: u64,
}

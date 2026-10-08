use eframe::egui;
use poker_tournament::PokerGuiApp;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 780.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("Texas Hold'em Deep CFR Tournament Simulator"),
        ..Default::default()
    };

    eframe::run_native(
        "Texas Hold'em Deep CFR Tournament",
        native_options,
        Box::new(|_cc| Ok(Box::new(PokerGuiApp::new(6)))),
    )
}

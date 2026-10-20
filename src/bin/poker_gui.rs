use eframe::egui;
use poker_tournament::PokerNowApp;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 840.0])
            .with_min_inner_size([960.0, 640.0])
            .with_title("PokerNow Deep CFR Tournament Simulator (8-Max)"),
        ..Default::default()
    };

    eframe::run_native(
        "PokerNow Deep CFR Tournament",
        native_options,
        Box::new(|_cc| Ok(Box::new(PokerNowApp::new(8)))),
    )
}

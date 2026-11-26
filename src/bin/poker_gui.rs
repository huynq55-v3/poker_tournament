use eframe::egui;
use poker_tournament::PokerNowApp;

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("PokerNow Deep CFR GUI Simulator (2-10 Max)");
        println!("Usage: cargo run --release --bin poker_gui -- [--lounge|--cash] [--seats <2..=10>]");
        return Ok(());
    }

    let is_lounge = args.iter().any(|a| a == "--lounge" || a == "--cash");
    let default_seats = if is_lounge { 10usize } else { 8usize };
    let seats = args
        .iter()
        .position(|a| a == "--seats" || a == "--players")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_seats)
        .clamp(2, poker_tournament::environment::MAX_PLAYERS);

    let title = if is_lounge {
        format!("PokerNow Deep CFR Lounge Cash ({}-Max)", seats)
    } else {
        format!("PokerNow Deep CFR Tournament ({}-Max)", seats)
    };

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 840.0])
            .with_min_inner_size([960.0, 640.0])
            .with_title(title),
        ..Default::default()
    };

    eframe::run_native(
        "PokerNow Deep CFR Simulator",
        native_options,
        Box::new(move |_cc| Ok(Box::new(PokerNowApp::new_with_mode(seats, is_lounge)))),
    )
}

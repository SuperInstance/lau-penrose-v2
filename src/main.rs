use lau_penrose_v2::{PenroseDetector, PenrosePredictor};

fn main() {
    println!("=== Penrose Correlation Engine v2 ===\n");

    let mut det = PenroseDetector::new(3, 0.3);

    // Feed correlated rooms
    for i in 0..100 {
        let t = i as f64;
        det.observe("room_alpha", i, t.sin());
        det.observe("room_beta", i, t.sin() * 0.95 + 0.05);
        det.observe("room_gamma", i, -t.sin());
        det.observe("room_delta", i, t.cos());
    }

    let matrix = det.compute_correlations();
    println!("Correlation matrix ({} rooms):", matrix.rooms.len());
    for room in &matrix.rooms {
        for other in &matrix.rooms {
            if let Some(v) = matrix.get(room, other) {
                print!("{:8.4}", v);
            }
        }
        println!();
    }

    let splines = det.detect_splines();
    println!("\nDetected {} splines:", splines.len());
    for s in &splines {
        println!(
            "  {} ↔ {} | coeff={:.4} type={:?} conf={:.4}",
            s.room_a, s.room_b, s.coefficient, s.spline_type, s.confidence
        );
    }

    let predictor = PenrosePredictor::new(det);
    let preds = predictor.predict_all();
    println!("\nPredictions:");
    for p in preds.values() {
        println!(
            "  {} → {:.4} (conf={:.4}, method={})",
            p.room_id, p.predicted_value, p.confidence, p.method
        );
    }
}

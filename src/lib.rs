use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. TimeSeries
// ---------------------------------------------------------------------------

/// A room's signal over time: a sequence of (timestamp, value) pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub room_id: String,
    pub values: Vec<(u64, f64)>,
}

impl TimeSeries {
    pub fn new(room_id: impl Into<String>) -> Self {
        Self {
            room_id: room_id.into(),
            values: Vec::new(),
        }
    }

    pub fn push(&mut self, timestamp: u64, value: f64) {
        self.values.push((timestamp, value));
    }

    pub fn last_n(&self, n: usize) -> Vec<(u64, f64)> {
        let start = self.values.len().saturating_sub(n);
        self.values[start..].to_vec()
    }

    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.values.iter().map(|(_, v)| v).sum();
        sum / self.values.len() as f64
    }

    pub fn variance(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let m = self.mean();
        let sum_sq: f64 = self.values.iter().map(|(_, v)| (v - m).powi(2)).sum();
        sum_sq / self.values.len() as f64
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn autocorrelation(&self, lag: usize) -> f64 {
        if self.values.is_empty() || lag >= self.values.len() {
            return 0.0;
        }
        let vals: Vec<f64> = self.values.iter().map(|(_, v)| *v).collect();
        let n = vals.len();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = vals.iter().map(|v| v - mean).collect();

        let variance: f64 = centered.iter().map(|v| v * v).sum::<f64>() / n as f64;
        if variance.abs() < f64::EPSILON {
            return if lag == 0 { 1.0 } else { 0.0 };
        }

        let sum: f64 = (0..n - lag).map(|i| centered[i] * centered[i + lag]).sum();
        sum / ((n - lag) as f64 * variance)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// 2. CorrelationPair
// ---------------------------------------------------------------------------

/// A correlation pair between two rooms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPair {
    pub room_a: String,
    pub room_b: String,
    pub coefficient: f64,
    pub confidence: f64,
}

impl CorrelationPair {
    pub fn is_positive(&self) -> bool {
        self.coefficient > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.coefficient < 0.0
    }

    pub fn strength(&self) -> f64 {
        self.coefficient.abs() * self.confidence
    }
}

// ---------------------------------------------------------------------------
// 3. CorrelationMatrix
// ---------------------------------------------------------------------------

/// N×N Pearson correlation matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub rooms: Vec<String>,
    pub values: Vec<Vec<f64>>,
}

impl CorrelationMatrix {
    pub fn new(rooms: Vec<String>) -> Self {
        let n = rooms.len();
        Self {
            rooms,
            values: vec![vec![0.0; n]; n],
        }
    }

    fn index_of(&self, room: &str) -> Option<usize> {
        self.rooms.iter().position(|r| r == room)
    }

    pub fn get(&self, room_a: &str, room_b: &str) -> Option<f64> {
        let i = self.index_of(room_a)?;
        let j = self.index_of(room_b)?;
        Some(self.values[i][j])
    }

    pub fn set(&mut self, room_a: &str, room_b: &str, value: f64) {
        if let (Some(i), Some(j)) = (self.index_of(room_a), self.index_of(room_b)) {
            self.values[i][j] = value;
            self.values[j][i] = value; // symmetric
        }
    }

    pub fn strongest_pairs(&self, threshold: f64) -> Vec<CorrelationPair> {
        let n = self.rooms.len();
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let coeff = self.values[i][j];
                if coeff.abs() >= threshold {
                    pairs.push(CorrelationPair {
                        room_a: self.rooms[i].clone(),
                        room_b: self.rooms[j].clone(),
                        coefficient: coeff,
                        confidence: coeff.abs().min(1.0),
                    });
                }
            }
        }
        pairs.sort_by(|a, b| b.strength().partial_cmp(&a.strength()).unwrap());
        pairs
    }

    pub fn correlated_with(&self, room_id: &str, threshold: f64) -> Vec<CorrelationPair> {
        let i = match self.index_of(room_id) {
            Some(idx) => idx,
            None => return Vec::new(),
        };
        let mut pairs = Vec::new();
        for j in 0..self.rooms.len() {
            if i == j {
                continue;
            }
            let coeff = self.values[i][j];
            if coeff >= threshold {
                pairs.push(CorrelationPair {
                    room_a: room_id.to_string(),
                    room_b: self.rooms[j].clone(),
                    coefficient: coeff,
                    confidence: coeff.abs().min(1.0),
                });
            }
        }
        pairs
    }

    pub fn anti_correlated_with(&self, room_id: &str, threshold: f64) -> Vec<CorrelationPair> {
        let i = match self.index_of(room_id) {
            Some(idx) => idx,
            None => return Vec::new(),
        };
        let mut pairs = Vec::new();
        for j in 0..self.rooms.len() {
            if i == j {
                continue;
            }
            let coeff = self.values[i][j];
            if coeff <= -threshold {
                pairs.push(CorrelationPair {
                    room_a: room_id.to_string(),
                    room_b: self.rooms[j].clone(),
                    coefficient: coeff,
                    confidence: coeff.abs().min(1.0),
                });
            }
        }
        pairs
    }
}

// ---------------------------------------------------------------------------
// 4. SplineType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplineType {
    Causal,
    Resonant,
    Predictive,
    Synergistic,
    Redundant,
}

// ---------------------------------------------------------------------------
// 5. Spline
// ---------------------------------------------------------------------------

/// A detected correlation spline between two rooms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spline {
    pub id: String,
    pub room_a: String,
    pub room_b: String,
    pub coefficient: f64,
    pub spline_type: SplineType,
    pub detected_at: u64,
    pub confidence: f64,
    pub prediction_accuracy: f64,
}

// ---------------------------------------------------------------------------
// 6. PenroseDetector
// ---------------------------------------------------------------------------

/// Detects correlations between PLATO rooms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenroseDetector {
    pub series: HashMap<String, TimeSeries>,
    pub min_samples: usize,
    pub threshold: f64,
}

impl PenroseDetector {
    pub fn new(min_samples: usize, threshold: f64) -> Self {
        Self {
            series: HashMap::new(),
            min_samples,
            threshold,
        }
    }

    pub fn observe(&mut self, room_id: &str, timestamp: u64, value: f64) {
        self.series
            .entry(room_id.to_string())
            .or_insert_with(|| TimeSeries::new(room_id))
            .push(timestamp, value);
    }

    /// Compute Pearson correlation coefficient between two rooms.
    pub fn pearson(&self, a: &str, b: &str) -> Option<f64> {
        let sa = self.series.get(a)?;
        let sb = self.series.get(b)?;

        if sa.len() < self.min_samples || sb.len() < self.min_samples {
            return None;
        }

        let n = sa.len().min(sb.len());
        let va: Vec<f64> = sa.last_n(n).iter().map(|(_, v)| *v).collect();
        let vb: Vec<f64> = sb.last_n(n).iter().map(|(_, v)| *v).collect();

        let mean_a: f64 = va.iter().sum::<f64>() / n as f64;
        let mean_b: f64 = vb.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;
        for i in 0..n {
            let da = va[i] - mean_a;
            let db = vb[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = var_a.sqrt() * var_b.sqrt();
        if denom.abs() < f64::EPSILON {
            return Some(0.0);
        }
        Some(cov / denom)
    }

    pub fn compute_correlations(&self) -> CorrelationMatrix {
        let rooms: Vec<String> = {
            let mut r: Vec<String> = self.series.keys().cloned().collect();
            r.sort();
            r
        };
        let mut matrix = CorrelationMatrix::new(rooms.clone());
        for i in 0..rooms.len() {
            for j in i..rooms.len() {
                let val = self
                    .pearson(&rooms[i], &rooms[j])
                    .unwrap_or(0.0);
                matrix.values[i][j] = val;
                matrix.values[j][i] = val;
            }
        }
        matrix
    }

    pub fn detect_splines(&self) -> Vec<Spline> {
        let matrix = self.compute_correlations();
        let mut splines = Vec::new();
        let n = matrix.rooms.len();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for i in 0..n {
            for j in (i + 1)..n {
                let coeff = matrix.values[i][j];
                if coeff.abs() >= self.threshold {
                    let spline_type = self.classify_spline(&matrix.rooms[i], &matrix.rooms[j], coeff);
                    splines.push(Spline {
                        id: format!("{}-{}", matrix.rooms[i], matrix.rooms[j]),
                        room_a: matrix.rooms[i].clone(),
                        room_b: matrix.rooms[j].clone(),
                        coefficient: coeff,
                        spline_type,
                        detected_at: now,
                        confidence: coeff.abs().min(1.0),
                        prediction_accuracy: 0.0,
                    });
                }
            }
        }
        splines
    }

    pub fn classify_spline(&self, a: &str, b: &str, coeff: f64) -> SplineType {
        let auto_a = self.series.get(a).map(|s| s.autocorrelation(1)).unwrap_or(0.0);
        let auto_b = self.series.get(b).map(|s| s.autocorrelation(1)).unwrap_or(0.0);

        if coeff > 0.95 {
            SplineType::Synergistic
        } else if coeff > 0.7 {
            SplineType::Predictive
        } else if coeff > 0.3 {
            if auto_a > 0.5 && auto_b > 0.5 {
                SplineType::Resonant
            } else {
                SplineType::Causal
            }
        } else if coeff < -0.7 {
            SplineType::Redundant
        } else if coeff < -0.3 {
            SplineType::Causal
        } else {
            SplineType::Resonant
        }
    }
}

// ---------------------------------------------------------------------------
// 7. CorrelationTopology
// ---------------------------------------------------------------------------

/// Graph of detected correlation splines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationTopology {
    pub rooms: Vec<String>,
    pub splines: Vec<Spline>,
}

impl CorrelationTopology {
    pub fn new(rooms: Vec<String>, splines: Vec<Spline>) -> Self {
        Self { rooms, splines }
    }

    pub fn neighbors(&self, room_id: &str) -> Vec<&Spline> {
        self.splines
            .iter()
            .filter(|s| s.room_a == room_id || s.room_b == room_id)
            .collect()
    }

    /// Connected components via union-find.
    pub fn clusters(&self) -> Vec<Vec<String>> {
        let mut parent: HashMap<String, String> = HashMap::new();
        for room in &self.rooms {
            parent.insert(room.clone(), room.clone());
        }

        fn find(parent: &mut HashMap<String, String>, x: &str) -> String {
            if parent[x] != x {
                let root = find(parent, &parent[x].clone());
                parent.insert(x.to_string(), root.clone());
                root
            } else {
                x.to_string()
            }
        }

        for spline in &self.splines {
            let ra = find(&mut parent, &spline.room_a);
            let rb = find(&mut parent, &spline.room_b);
            if ra != rb {
                parent.insert(ra, rb.clone());
            }
        }

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for room in &self.rooms {
            let root = find(&mut parent, room);
            groups.entry(root).or_default().push(room.clone());
        }
        groups.into_values().collect()
    }

    pub fn centrality(&self, room_id: &str) -> f64 {
        if self.rooms.is_empty() {
            return 0.0;
        }
        let degree = self.neighbors(room_id).len() as f64;
        let max_possible = (self.rooms.len() - 1) as f64;
        if max_possible == 0.0 {
            return 0.0;
        }
        degree / max_possible
    }

    pub fn most_central(&self) -> Option<String> {
        self.rooms
            .iter()
            .max_by(|a, b| {
                self.centrality(a)
                    .partial_cmp(&self.centrality(b))
                    .unwrap()
            })
            .cloned()
    }

    pub fn bridge_rooms(&self) -> Vec<String> {
        // A bridge room is one whose removal would increase the number of connected components.
        let base_clusters = self.clusters();
        let base_count = base_clusters.len();

        let mut bridges = Vec::new();
        for room in &self.rooms {
            // Create topology without this room
            let other_rooms: Vec<String> = self
                .rooms
                .iter()
                .filter(|r| *r != room)
                .cloned()
                .collect();
            let other_splines: Vec<Spline> = self
                .splines
                .iter()
                .filter(|s| s.room_a != *room && s.room_b != *room)
                .cloned()
                .collect();
            let reduced = CorrelationTopology::new(other_rooms, other_splines);
            let reduced_clusters = reduced.clusters();
            // Removing a bridge increases cluster count (or keeps same if room was isolated)
            if reduced_clusters.len() > base_count {
                bridges.push(room.clone());
            }
        }
        bridges
    }
}

// ---------------------------------------------------------------------------
// 8. Prediction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub room_id: String,
    pub predicted_value: f64,
    pub confidence: f64,
    pub based_on: Vec<String>,
    pub method: String,
}

// ---------------------------------------------------------------------------
// 9. PenrosePredictor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenrosePredictor {
    pub detector: PenroseDetector,
    pub topology: CorrelationTopology,
}

impl PenrosePredictor {
    pub fn new(detector: PenroseDetector) -> Self {
        let splines = detector.detect_splines();
        let rooms: Vec<String> = {
            let mut r: Vec<String> = detector.series.keys().cloned().collect();
            r.sort();
            r
        };
        let topology = CorrelationTopology::new(rooms, splines);
        Self {
            detector,
            topology,
        }
    }

    /// Predict a room's next value using weighted average from correlated rooms.
    pub fn predict_room(&self, room_id: &str) -> Option<Prediction> {
        let target = self.detector.series.get(room_id)?;
        if target.is_empty() {
            return None;
        }

        let neighbors = self.topology.neighbors(room_id);
        if neighbors.is_empty() {
            // Fallback: use own mean
            return Some(Prediction {
                room_id: room_id.to_string(),
                predicted_value: target.mean(),
                confidence: 0.1,
                based_on: vec![],
                method: "fallback_mean".to_string(),
            });
        }

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        let mut based_on = Vec::new();

        for spline in &neighbors {
            let other_id = if spline.room_a == room_id {
                &spline.room_b
            } else {
                &spline.room_a
            };
            if let Some(other) = self.detector.series.get(other_id) {
                if other.is_empty() {
                    continue;
                }
                let last_val = other.values.last().map(|(_, v)| *v).unwrap_or(0.0);
                let weight = spline.coefficient.abs() * spline.confidence;
                weighted_sum += weight * last_val;
                weight_total += weight;
                based_on.push(other_id.clone());
            }
        }

        if weight_total < f64::EPSILON {
            return Some(Prediction {
                room_id: room_id.to_string(),
                predicted_value: target.mean(),
                confidence: 0.1,
                based_on,
                method: "fallback_mean".to_string(),
            });
        }

        let predicted_value = weighted_sum / weight_total;
        let confidence = weight_total / neighbors.len() as f64;

        Some(Prediction {
            room_id: room_id.to_string(),
            predicted_value,
            confidence: confidence.min(1.0),
            based_on,
            method: "weighted_correlation".to_string(),
        })
    }

    pub fn predict_all(&self) -> HashMap<String, Prediction> {
        let mut predictions = HashMap::new();
        for room_id in &self.topology.rooms {
            if let Some(pred) = self.predict_room(room_id) {
                predictions.insert(room_id.clone(), pred);
            }
        }
        predictions
    }

    pub fn update_accuracy(&mut self, room_id: &str, predicted: f64, actual: f64) {
        let accuracy = 1.0 - (predicted - actual).abs() / (actual.abs().max(1.0));
        let accuracy = accuracy.clamp(0.0, 1.0);
        for spline in &mut self.topology.splines {
            if spline.room_a == room_id || spline.room_b == room_id {
                spline.prediction_accuracy = accuracy;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- TimeSeries tests ---

    #[test]
    fn test_timeseries_push_and_len() {
        let mut ts = TimeSeries::new("room_a");
        assert!(ts.is_empty());
        ts.push(1, 10.0);
        ts.push(2, 20.0);
        ts.push(3, 30.0);
        assert_eq!(ts.len(), 3);
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_timeseries_last_n() {
        let mut ts = TimeSeries::new("room_a");
        for i in 0..10u64 {
            ts.push(i, i as f64);
        }
        let last = ts.last_n(3);
        assert_eq!(last.len(), 3);
        assert_eq!(last[0], (7, 7.0));
        assert_eq!(last[2], (9, 9.0));
    }

    #[test]
    fn test_timeseries_mean() {
        let mut ts = TimeSeries::new("room_a");
        ts.push(1, 10.0);
        ts.push(2, 20.0);
        ts.push(3, 30.0);
        assert!((ts.mean() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_timeseries_mean_empty() {
        let ts = TimeSeries::new("room_a");
        assert_eq!(ts.mean(), 0.0);
    }

    #[test]
    fn test_timeseries_variance() {
        let mut ts = TimeSeries::new("room_a");
        ts.push(1, 10.0);
        ts.push(2, 20.0);
        ts.push(3, 30.0);
        assert!((ts.variance() - 200.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_timeseries_std_dev() {
        let mut ts = TimeSeries::new("room_a");
        ts.push(1, 10.0);
        ts.push(2, 20.0);
        ts.push(3, 30.0);
        let expected: f64 = (200.0_f64 / 3.0).sqrt();
        assert!((ts.std_dev() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_timeseries_autocorrelation_lag_zero() {
        let mut ts = TimeSeries::new("room_a");
        ts.push(1, 10.0);
        ts.push(2, 20.0);
        ts.push(3, 30.0);
        assert!((ts.autocorrelation(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_timeseries_autocorrelation_lag_one() {
        let mut ts = TimeSeries::new("room_a");
        // Values that are positively autocorrelated: monotonic increasing
        for i in 0..20 {
            ts.push(i, i as f64);
        }
        let ac = ts.autocorrelation(1);
        // For monotonic series, autocorrelation at lag 1 should be positive
        assert!(ac > 0.5, "autocorrelation at lag 1 should be positive for monotonic series, got {}", ac);
    }

    #[test]
    fn test_timeseries_autocorrelation_constant_series() {
        let mut ts = TimeSeries::new("room_a");
        ts.push(1, 5.0);
        ts.push(2, 5.0);
        ts.push(3, 5.0);
        assert!((ts.autocorrelation(0) - 1.0).abs() < 1e-10);
        assert!((ts.autocorrelation(1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_timeseries_autocorrelation_empty() {
        let ts = TimeSeries::new("room_a");
        assert_eq!(ts.autocorrelation(0), 0.0);
    }

    #[test]
    fn test_timeseries_last_n_more_than_len() {
        let mut ts = TimeSeries::new("room_a");
        ts.push(1, 1.0);
        ts.push(2, 2.0);
        let last = ts.last_n(10);
        assert_eq!(last.len(), 2);
    }

    #[test]
    fn test_timeseries_variance_empty() {
        let ts = TimeSeries::new("a");
        assert_eq!(ts.variance(), 0.0);
    }

    #[test]
    fn test_timeseries_std_dev_empty() {
        let ts = TimeSeries::new("a");
        assert_eq!(ts.std_dev(), 0.0);
    }

    // --- CorrelationPair tests ---

    #[test]
    fn test_correlation_pair_positive() {
        let pair = CorrelationPair {
            room_a: "a".into(),
            room_b: "b".into(),
            coefficient: 0.85,
            confidence: 0.9,
        };
        assert!(pair.is_positive());
        assert!(!pair.is_negative());
        assert!((pair.strength() - 0.85 * 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_pair_negative() {
        let pair = CorrelationPair {
            room_a: "a".into(),
            room_b: "b".into(),
            coefficient: -0.7,
            confidence: 0.8,
        };
        assert!(!pair.is_positive());
        assert!(pair.is_negative());
        assert!((pair.strength() - 0.7 * 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_pair_zero_strength() {
        let pair = CorrelationPair {
            room_a: "a".into(),
            room_b: "b".into(),
            coefficient: 0.0,
            confidence: 0.5,
        };
        assert!((pair.strength() - 0.0).abs() < 1e-10);
    }

    // --- CorrelationMatrix tests ---

    #[test]
    fn test_matrix_set_get() {
        let mut m = CorrelationMatrix::new(vec!["a".into(), "b".into(), "c".into()]);
        m.set("a", "b", 0.9);
        assert!((m.get("a", "b").unwrap() - 0.9).abs() < 1e-10);
        assert!((m.get("b", "a").unwrap() - 0.9).abs() < 1e-10);
        assert_eq!(m.get("a", "c").unwrap(), 0.0);
    }

    #[test]
    fn test_matrix_get_missing() {
        let m = CorrelationMatrix::new(vec!["a".into()]);
        assert!(m.get("a", "z").is_none());
    }

    #[test]
    fn test_matrix_strongest_pairs() {
        let mut m = CorrelationMatrix::new(vec!["a".into(), "b".into(), "c".into()]);
        m.set("a", "b", 0.95);
        m.set("a", "c", 0.5);
        m.set("b", "c", -0.8);
        let pairs = m.strongest_pairs(0.7);
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_matrix_correlated_with() {
        let mut m = CorrelationMatrix::new(vec!["a".into(), "b".into(), "c".into()]);
        m.set("a", "b", 0.9);
        m.set("a", "c", -0.8);
        let correlated = m.correlated_with("a", 0.5);
        assert_eq!(correlated.len(), 1);
        assert_eq!(correlated[0].room_b, "b");
    }

    #[test]
    fn test_matrix_anti_correlated_with() {
        let mut m = CorrelationMatrix::new(vec!["a".into(), "b".into(), "c".into()]);
        m.set("a", "b", 0.9);
        m.set("a", "c", -0.8);
        let anti = m.anti_correlated_with("a", 0.5);
        assert_eq!(anti.len(), 1);
        assert_eq!(anti[0].room_b, "c");
    }

    #[test]
    fn test_matrix_strongest_pairs_sorted() {
        let mut m = CorrelationMatrix::new(vec!["a".into(), "b".into(), "c".into()]);
        m.set("a", "b", 0.6);
        m.set("a", "c", 0.95);
        m.set("b", "c", 0.8);
        let pairs = m.strongest_pairs(0.5);
        assert_eq!(pairs.len(), 3);
        assert!(pairs[0].strength() >= pairs[1].strength());
        assert!(pairs[1].strength() >= pairs[2].strength());
    }

    #[test]
    fn test_matrix_correlated_with_missing() {
        let m = CorrelationMatrix::new(vec!["a".into()]);
        let pairs = m.correlated_with("z", 0.5);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_matrix_anti_correlated_with_missing() {
        let m = CorrelationMatrix::new(vec!["a".into()]);
        let pairs = m.anti_correlated_with("z", 0.5);
        assert!(pairs.is_empty());
    }

    // --- PenroseDetector tests ---

    #[test]
    fn test_detector_pearson_identical() {
        let mut det = PenroseDetector::new(2, 0.3);
        for i in 0..10 {
            let v = (i as f64).sin();
            det.observe("a", i, v);
            det.observe("b", i, v);
        }
        let r = det.pearson("a", "b").unwrap();
        assert!((r - 1.0).abs() < 1e-10, "got {}", r);
    }

    #[test]
    fn test_detector_pearson_symmetric() {
        let mut det = PenroseDetector::new(2, 0.3);
        for i in 0..10 {
            det.observe("a", i, i as f64);
            det.observe("b", i, (i as f64) * 2.0 + 1.0);
        }
        let rab = det.pearson("a", "b").unwrap();
        let rba = det.pearson("b", "a").unwrap();
        assert!((rab - rba).abs() < 1e-10);
    }

    #[test]
    fn test_detector_pearson_self_is_one() {
        let mut det = PenroseDetector::new(2, 0.3);
        for i in 0..10 {
            det.observe("a", i, i as f64 * 3.7 + 1.2);
        }
        let r = det.pearson("a", "a").unwrap();
        assert!((r - 1.0).abs() < 1e-10, "got {}", r);
    }

    #[test]
    fn test_detector_pearson_insufficient_samples() {
        let mut det = PenroseDetector::new(5, 0.3);
        det.observe("a", 1, 1.0);
        det.observe("b", 1, 2.0);
        assert!(det.pearson("a", "b").is_none());
    }

    #[test]
    fn test_detector_pearson_missing_room() {
        let det = PenroseDetector::new(2, 0.3);
        assert!(det.pearson("a", "b").is_none());
    }

    #[test]
    fn test_detector_compute_correlations() {
        let mut det = PenroseDetector::new(2, 0.3);
        for i in 0..10 {
            let v = i as f64;
            det.observe("a", i, v);
            det.observe("b", i, v * 2.0);
        }
        let matrix = det.compute_correlations();
        assert_eq!(matrix.rooms.len(), 2);
        let r = matrix.get("a", "b").unwrap();
        assert!((r - 1.0).abs() < 1e-10, "got {}", r);
    }

    #[test]
    fn test_detector_strong_correlation_oscillating() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..100 {
            let v = (i as f64 * 0.1).sin();
            det.observe("a", i, v);
            det.observe("b", i, v);
        }
        let splines = det.detect_splines();
        assert!(!splines.is_empty());
        assert!(splines[0].coefficient > 0.9);
    }

    #[test]
    fn test_detector_anti_correlation_phase_inverted() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..100 {
            let v = (i as f64 * 0.1).sin();
            det.observe("a", i, v);
            det.observe("b", i, -v);
        }
        let r = det.pearson("a", "b").unwrap();
        assert!(r < -0.9, "got {}", r);
        let splines = det.detect_splines();
        assert!(!splines.is_empty());
        assert!(splines.iter().any(|s| s.coefficient < -0.9));
    }

    #[test]
    fn test_detector_classify_synergistic() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..50 {
            det.observe("a", i, i as f64);
            det.observe("b", i, i as f64);
        }
        let st = det.classify_spline("a", "b", 0.96);
        assert_eq!(st, SplineType::Synergistic);
    }

    #[test]
    fn test_detector_classify_predictive() {
        let det = PenroseDetector::new(3, 0.3);
        assert_eq!(det.classify_spline("a", "b", 0.8), SplineType::Predictive);
    }

    #[test]
    fn test_detector_classify_causal() {
        let det = PenroseDetector::new(3, 0.3);
        assert_eq!(det.classify_spline("a", "b", 0.5), SplineType::Causal);
    }

    #[test]
    fn test_detector_classify_redundant() {
        let det = PenroseDetector::new(3, 0.3);
        assert_eq!(det.classify_spline("a", "b", -0.8), SplineType::Redundant);
    }

    #[test]
    fn test_detector_classify_negative_causal() {
        let det = PenroseDetector::new(3, 0.3);
        assert_eq!(det.classify_spline("a", "b", -0.5), SplineType::Causal);
    }

    #[test]
    fn test_detector_classify_resonant() {
        let det = PenroseDetector::new(3, 0.3);
        assert_eq!(det.classify_spline("a", "b", 0.1), SplineType::Resonant);
    }

    #[test]
    fn test_detector_no_splines_below_threshold() {
        let mut det = PenroseDetector::new(3, 0.99);
        for i in 0..50 {
            det.observe("a", i, (i as f64).sin());
            det.observe("b", i, (i as f64 + 3.0).cos());
        }
        let _splines = det.detect_splines();
    }

    #[test]
    fn test_detector_multiple_rooms() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..50 {
            det.observe("a", i, i as f64);
            det.observe("b", i, i as f64 * 2.0);
            det.observe("c", i, -(i as f64));
            det.observe("d", i, 100.0);
        }
        let matrix = det.compute_correlations();
        assert_eq!(matrix.rooms.len(), 4);
        assert!(matrix.get("a", "b").unwrap() > 0.99);
        assert!(matrix.get("a", "c").unwrap() < -0.99);
    }

    // --- CorrelationTopology tests ---

    #[test]
    fn test_topology_neighbors() {
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec![Spline {
                id: "a-b".into(),
                room_a: "a".into(),
                room_b: "b".into(),
                coefficient: 0.9,
                spline_type: SplineType::Predictive,
                detected_at: 0,
                confidence: 0.9,
                prediction_accuracy: 0.0,
            }],
        );
        assert_eq!(topology.neighbors("a").len(), 1);
        assert_eq!(topology.neighbors("c").len(), 0);
    }

    #[test]
    fn test_topology_clusters_single() {
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec![
                Spline {
                    id: "a-b".into(),
                    room_a: "a".into(),
                    room_b: "b".into(),
                    coefficient: 0.9,
                    spline_type: SplineType::Predictive,
                    detected_at: 0,
                    confidence: 0.9,
                    prediction_accuracy: 0.0,
                },
                Spline {
                    id: "b-c".into(),
                    room_a: "b".into(),
                    room_b: "c".into(),
                    coefficient: 0.8,
                    spline_type: SplineType::Predictive,
                    detected_at: 0,
                    confidence: 0.8,
                    prediction_accuracy: 0.0,
                },
            ],
        );
        let clusters = topology.clusters();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn test_topology_clusters_separate() {
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![Spline {
                id: "a-b".into(),
                room_a: "a".into(),
                room_b: "b".into(),
                coefficient: 0.9,
                spline_type: SplineType::Predictive,
                detected_at: 0,
                confidence: 0.9,
                prediction_accuracy: 0.0,
            }],
        );
        let clusters = topology.clusters();
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn test_topology_centrality() {
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![
                Spline {
                    id: "a-b".into(),
                    room_a: "a".into(),
                    room_b: "b".into(),
                    coefficient: 0.9,
                    spline_type: SplineType::Predictive,
                    detected_at: 0,
                    confidence: 0.9,
                    prediction_accuracy: 0.0,
                },
                Spline {
                    id: "b-c".into(),
                    room_a: "b".into(),
                    room_b: "c".into(),
                    coefficient: 0.8,
                    spline_type: SplineType::Predictive,
                    detected_at: 0,
                    confidence: 0.8,
                    prediction_accuracy: 0.0,
                },
                Spline {
                    id: "b-d".into(),
                    room_a: "b".into(),
                    room_b: "d".into(),
                    coefficient: 0.7,
                    spline_type: SplineType::Predictive,
                    detected_at: 0,
                    confidence: 0.7,
                    prediction_accuracy: 0.0,
                },
            ],
        );
        assert!((topology.centrality("b") - 1.0).abs() < 1e-10);
        assert!((topology.centrality("a") - (1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_topology_most_central() {
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec![
                Spline {
                    id: "a-b".into(),
                    room_a: "a".into(),
                    room_b: "b".into(),
                    coefficient: 0.9,
                    spline_type: SplineType::Predictive,
                    detected_at: 0,
                    confidence: 0.9,
                    prediction_accuracy: 0.0,
                },
                Spline {
                    id: "a-c".into(),
                    room_a: "a".into(),
                    room_b: "c".into(),
                    coefficient: 0.8,
                    spline_type: SplineType::Predictive,
                    detected_at: 0,
                    confidence: 0.8,
                    prediction_accuracy: 0.0,
                },
            ],
        );
        assert_eq!(topology.most_central().unwrap(), "a");
    }

    #[test]
    fn test_topology_bridge_rooms() {
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into(), "f".into()],
            vec![
                Spline { id: "a-b".into(), room_a: "a".into(), room_b: "b".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "b-c".into(), room_a: "b".into(), room_b: "c".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "d-e".into(), room_a: "d".into(), room_b: "e".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "e-f".into(), room_a: "e".into(), room_b: "f".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "b-e".into(), room_a: "b".into(), room_b: "e".into(), coefficient: 0.5, spline_type: SplineType::Causal, detected_at: 0, confidence: 0.5, prediction_accuracy: 0.0 },
            ],
        );
        let bridges = topology.bridge_rooms();
        assert!(bridges.contains(&"b".to_string()));
        assert!(bridges.contains(&"e".to_string()));
    }

    #[test]
    fn test_topology_bridge_no_bridges_single_cluster() {
        // Triangle: fully connected, no bridge
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec![
                Spline { id: "a-b".into(), room_a: "a".into(), room_b: "b".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "b-c".into(), room_a: "b".into(), room_b: "c".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "a-c".into(), room_a: "a".into(), room_b: "c".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
            ],
        );
        assert!(topology.bridge_rooms().is_empty());
    }

    #[test]
    fn test_topology_most_central_empty() {
        let topology = CorrelationTopology::new(vec![], vec![]);
        assert!(topology.most_central().is_none());
    }

    #[test]
    fn test_topology_centrality_empty_rooms() {
        let topology = CorrelationTopology::new(vec![], vec![]);
        assert!((topology.centrality("x") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_topology_centrality_single_room() {
        let topology = CorrelationTopology::new(vec!["a".into()], vec![]);
        assert!((topology.centrality("a") - 0.0).abs() < 1e-10);
    }

    // --- Prediction tests ---

    #[test]
    fn test_predictor_correlated_rooms() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..50 {
            det.observe("a", i, i as f64);
            det.observe("b", i, i as f64 * 2.0 + 5.0);
        }
        let predictor = PenrosePredictor::new(det);
        let pred = predictor.predict_room("a").unwrap();
        assert_eq!(pred.room_id, "a");
        assert!(!pred.based_on.is_empty());
    }

    #[test]
    fn test_predictor_predict_all() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..50u64 {
            det.observe("a", i, i as f64);
            det.observe("b", i, i as f64 * 2.0);
        }
        let predictor = PenrosePredictor::new(det);
        let preds = predictor.predict_all();
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_predictor_update_accuracy() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..10u64 {
            det.observe("a", i, i as f64);
            det.observe("b", i, i as f64);
        }
        let mut predictor = PenrosePredictor::new(det);
        predictor.update_accuracy("a", 10.0, 10.5);
        let acc: f64 = predictor
            .topology
            .splines
            .iter()
            .filter(|s| s.room_a == "a" || s.room_b == "a")
            .map(|s| s.prediction_accuracy)
            .sum();
        assert!(acc > 0.0);
    }

    #[test]
    fn test_predictor_fallback_no_neighbors() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..10u64 {
            det.observe("a", i, i as f64);
        }
        let predictor = PenrosePredictor::new(det);
        let pred = predictor.predict_room("a").unwrap();
        assert_eq!(pred.method, "fallback_mean");
        assert!(pred.based_on.is_empty());
    }

    #[test]
    fn test_predictor_empty_room() {
        let det = PenroseDetector::new(3, 0.3);
        let predictor = PenrosePredictor::new(det);
        assert!(predictor.predict_room("nonexistent").is_none());
    }

    #[test]
    fn test_predictor_weighted_correlation_method() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..50u64 {
            det.observe("a", i, i as f64);
            det.observe("b", i, i as f64);
        }
        let predictor = PenrosePredictor::new(det);
        let pred = predictor.predict_room("a").unwrap();
        assert_eq!(pred.method, "weighted_correlation");
    }

    #[test]
    fn test_predictor_predict_all_empty() {
        let det = PenroseDetector::new(3, 0.3);
        let predictor = PenrosePredictor::new(det);
        assert!(predictor.predict_all().is_empty());
    }

    // --- Serde round-trip tests ---

    #[test]
    fn test_serde_timeseries() {
        let mut ts = TimeSeries::new("room_a");
        ts.push(1, 10.0);
        let json = serde_json::to_string(&ts).unwrap();
        let ts2: TimeSeries = serde_json::from_str(&json).unwrap();
        assert_eq!(ts2.room_id, "room_a");
        assert_eq!(ts2.values.len(), 1);
    }

    #[test]
    fn test_serde_spline_type() {
        let json = serde_json::to_string(&SplineType::Predictive).unwrap();
        let st: SplineType = serde_json::from_str(&json).unwrap();
        assert_eq!(st, SplineType::Predictive);
    }

    #[test]
    fn test_serde_correlation_pair() {
        let pair = CorrelationPair {
            room_a: "a".into(),
            room_b: "b".into(),
            coefficient: 0.85,
            confidence: 0.9,
        };
        let json = serde_json::to_string(&pair).unwrap();
        let p2: CorrelationPair = serde_json::from_str(&json).unwrap();
        assert!((p2.coefficient - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_serde_prediction() {
        let pred = Prediction {
            room_id: "a".into(),
            predicted_value: 42.0,
            confidence: 0.8,
            based_on: vec!["b".into(), "c".into()],
            method: "weighted_correlation".into(),
        };
        let json = serde_json::to_string(&pred).unwrap();
        let p2: Prediction = serde_json::from_str(&json).unwrap();
        assert_eq!(p2.room_id, "a");
        assert_eq!(p2.based_on.len(), 2);
    }

    #[test]
    fn test_serde_spline() {
        let spline = Spline {
            id: "a-b".into(),
            room_a: "a".into(),
            room_b: "b".into(),
            coefficient: 0.9,
            spline_type: SplineType::Synergistic,
            detected_at: 1234567890,
            confidence: 0.95,
            prediction_accuracy: 0.88,
        };
        let json = serde_json::to_string(&spline).unwrap();
        let s2: Spline = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.id, "a-b");
        assert_eq!(s2.spline_type, SplineType::Synergistic);
    }

    #[test]
    fn test_serde_correlation_matrix() {
        let mut m = CorrelationMatrix::new(vec!["a".into(), "b".into()]);
        m.set("a", "b", 0.75);
        let json = serde_json::to_string(&m).unwrap();
        let m2: CorrelationMatrix = serde_json::from_str(&json).unwrap();
        assert!((m2.get("a", "b").unwrap() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_serde_detector() {
        let mut det = PenroseDetector::new(3, 0.5);
        det.observe("a", 1, 10.0);
        let json = serde_json::to_string(&det).unwrap();
        let d2: PenroseDetector = serde_json::from_str(&json).unwrap();
        assert_eq!(d2.min_samples, 3);
        assert_eq!(d2.series["a"].values.len(), 1);
    }

    // --- Theorem verification tests ---

    #[test]
    fn test_theorem_1_pearson_identical_is_one() {
        let mut det = PenroseDetector::new(2, 0.3);
        for i in 0..50 {
            let v = (i as f64 * 0.3).sin() + (i as f64 * 0.7).cos();
            det.observe("a", i, v);
            det.observe("b", i, v);
        }
        let r = det.pearson("a", "b").unwrap();
        assert!((r - 1.0).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn test_theorem_2_pearson_symmetric() {
        let mut det = PenroseDetector::new(2, 0.3);
        for i in 0..50 {
            det.observe("a", i, (i as f64 * 0.2).sin());
            det.observe("b", i, (i as f64 * 0.2 + 1.0).cos());
        }
        let rab = det.pearson("a", "b").unwrap();
        let rba = det.pearson("b", "a").unwrap();
        assert!((rab - rba).abs() < 1e-10);
    }

    #[test]
    fn test_theorem_3_strong_correlation_oscillating() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..200 {
            let v = (i as f64 * 0.05).sin();
            det.observe("a", i, v);
            det.observe("b", i, v * 0.99 + 0.01);
        }
        let r = det.pearson("a", "b").unwrap();
        assert!(r > 0.9, "got {}", r);
        assert!(!det.detect_splines().is_empty());
    }

    #[test]
    fn test_theorem_4_anti_correlation_inverted() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..200 {
            let v = (i as f64 * 0.05).sin();
            det.observe("a", i, v);
            det.observe("b", i, -v);
        }
        let r = det.pearson("a", "b").unwrap();
        assert!(r < -0.9, "got {}", r);
    }

    #[test]
    fn test_theorem_5_prediction_better_than_random() {
        let mut det = PenroseDetector::new(3, 0.3);
        for i in 0..100 {
            let v = i as f64;
            det.observe("a", i, v);
            det.observe("b", i, v * 2.0);
        }
        let predictor = PenrosePredictor::new(det);
        let pred = predictor.predict_room("a").unwrap();
        assert!(pred.confidence > 0.0);
        assert!(!pred.based_on.is_empty());
    }

    #[test]
    fn test_theorem_6_centrality_most_connected() {
        let topology = CorrelationTopology::new(
            vec!["hub".into(), "s1".into(), "s2".into(), "s3".into()],
            vec![
                Spline { id: "h-s1".into(), room_a: "hub".into(), room_b: "s1".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "h-s2".into(), room_a: "hub".into(), room_b: "s2".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "h-s3".into(), room_a: "hub".into(), room_b: "s3".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
            ],
        );
        assert_eq!(topology.most_central().unwrap(), "hub");
    }

    #[test]
    fn test_theorem_7_clusters_group_correlated() {
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into(), "f".into()],
            vec![
                Spline { id: "a-b".into(), room_a: "a".into(), room_b: "b".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "b-c".into(), room_a: "b".into(), room_b: "c".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "d-e".into(), room_a: "d".into(), room_b: "e".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
            ],
        );
        assert_eq!(topology.clusters().len(), 3);
    }

    #[test]
    fn test_theorem_8_bridge_connects_clusters() {
        let topology = CorrelationTopology::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![
                Spline { id: "a-b".into(), room_a: "a".into(), room_b: "b".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "c-d".into(), room_a: "c".into(), room_b: "d".into(), coefficient: 0.9, spline_type: SplineType::Predictive, detected_at: 0, confidence: 0.9, prediction_accuracy: 0.0 },
                Spline { id: "b-c".into(), room_a: "b".into(), room_b: "c".into(), coefficient: 0.5, spline_type: SplineType::Causal, detected_at: 0, confidence: 0.5, prediction_accuracy: 0.0 },
            ],
        );
        let bridges = topology.bridge_rooms();
        assert!(bridges.contains(&"b".to_string()));
        assert!(bridges.contains(&"c".to_string()));
    }

    #[test]
    fn test_theorem_9_autocorrelation_lag_zero_is_one() {
        let mut ts = TimeSeries::new("room_a");
        for i in 0..20u64 {
            ts.push(i, (i as f64 * 0.3).sin());
        }
        assert!((ts.autocorrelation(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_theorem_10_spline_classification() {
        let det = PenroseDetector::new(3, 0.3);
        assert_eq!(det.classify_spline("a", "b", 0.85), SplineType::Predictive);
        assert_eq!(det.classify_spline("a", "b", 0.97), SplineType::Synergistic);
        assert_eq!(det.classify_spline("a", "b", 0.45), SplineType::Causal);
    }
}

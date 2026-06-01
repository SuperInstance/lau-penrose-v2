# lau-penrose-v2

A **Penrose correlation engine** for detecting, classifying, and predicting correlations between PLATO rooms — built on Pearson correlation, autocorrelation, and graph-theoretic topology analysis.

## What This Does

Imagine you have dozens of rooms (sensors, agents, processes — "PLATO rooms"), each emitting a time-series signal. This library answers three questions:

1. **Which rooms are correlated?** — Compute pairwise Pearson correlation across all rooms and surface the strongest links.
2. **What kind of link is it?** — Classify each correlation as *Causal*, *Resonant*, *Predictive*, *Synergistic*, or *Redundant* based on coefficient magnitude and autocorrelation signatures.
3. **What happens next?** — Predict a room's next value using a weighted average of its correlated neighbours, then track prediction accuracy over time.

It also builds a **correlation topology** (graph) and can find connected clusters, degree centrality, and bridge rooms whose removal would split the network.

## Key Idea

The library treats correlation detection as a layered pipeline:

```
TimeSeries → PenroseDetector → CorrelationMatrix → Splines → CorrelationTopology → Predictions
```

Each layer builds on the previous one, and every data structure is serialisable (serde `Serialize`/`Deserialize`), so you can persist and reload state.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-penrose-v2 = "0.1"
```

Or use `cargo add`:

```bash
cargo add lau-penrose-v2
```

Requires **Rust 2021 edition** (1.56+).

## Quick Start

```rust
use lau_penrose_v2::{PenroseDetector, PenrosePredictor};

fn main() {
    // 1. Create a detector: require ≥3 samples per room, threshold |r| ≥ 0.3
    let mut det = PenroseDetector::new(3, 0.3);

    // 2. Feed time-series observations (room_id, timestamp, value)
    for i in 0..100u64 {
        let v = (i as f64 * 0.05).sin();
        det.observe("room_a", i, v);
        det.observe("room_b", i, v * 0.98 + 0.02); // almost identical
        det.observe("room_c", i, -v);               // anti-correlated
    }

    // 3. Detect correlation splines
    let splines = det.detect_splines();
    for s in &splines {
        println!("{} ↔ {}  coeff={:.3}  type={:?}",
                 s.room_a, s.room_b, s.coefficient, s.spline_type);
    }

    // 4. Build a predictor and predict next values
    let predictor = PenrosePredictor::new(det);
    for (room, pred) in &predictor.predict_all() {
        println!("{} → predicted={:.3}  confidence={:.3}  method={}",
                 room, pred.predicted_value, pred.confidence, pred.method);
    }
}
```

## API Reference

### `TimeSeries`

A named sequence of `(timestamp, value)` pairs for one room.

| Method | Description |
|--------|-------------|
| `new(room_id)` | Create an empty time series |
| `push(timestamp, value)` | Append an observation |
| `last_n(n)` | Return the last *n* points |
| `mean()`, `variance()`, `std_dev()` | Basic statistics |
| `autocorrelation(lag)` | Autocorrelation at given lag (lag 0 = 1.0) |
| `len()`, `is_empty()` | Length queries |

### `CorrelationPair`

A directional view of correlation between two rooms: `coefficient` (Pearson *r*), `confidence`, and `strength()` (= |r| × confidence).

### `CorrelationMatrix`

An N×N symmetric matrix of Pearson coefficients.

| Method | Description |
|--------|-------------|
| `new(rooms)` | Create a zeroed matrix |
| `get(a, b)` / `set(a, b, val)` | Access by room name |
| `strongest_pairs(threshold)` | All pairs with |r| ≥ threshold, sorted by strength |
| `correlated_with(room, threshold)` | Positive correlations for one room |
| `anti_correlated_with(room, threshold)` | Negative correlations for one room |

### `Spline` and `SplineType`

A detected correlation edge between two rooms, classified as one of:

| Variant | Meaning |
|---------|---------|
| `Causal` | Moderate correlation (0.3–0.7), possibly negative |
| `Resonant` | Weak/moderate correlation *and* both rooms have high autocorrelation |
| `Predictive` | Strong correlation (0.7–0.95) |
| `Synergistic` | Near-perfect correlation (>0.95) |
| `Redundant` | Strong anti-correlation (<−0.7) |

### `PenroseDetector`

The core engine. Holds a `HashMap<String, TimeSeries>` of all rooms.

| Method | Description |
|--------|-------------|
| `new(min_samples, threshold)` | Configure detection sensitivity |
| `observe(room, timestamp, value)` | Feed data |
| `pearson(a, b)` | Pearson *r* between two rooms |
| `compute_correlations()` | Full N×N `CorrelationMatrix` |
| `detect_splines()` | All splines exceeding threshold |
| `classify_spline(a, b, coeff)` | Determine `SplineType` for a pair |

### `CorrelationTopology`

A graph of rooms and splines.

| Method | Description |
|--------|-------------|
| `new(rooms, splines)` | Build from detected splines |
| `neighbors(room)` | All splines touching a room |
| `clusters()` | Connected components via union-find |
| `centrality(room)` | Degree centrality (fraction of other rooms connected) |
| `most_central()` | Room with highest centrality |
| `bridge_rooms()` | Rooms whose removal increases cluster count |

### `PenrosePredictor`

Predicts a room's next value from its correlated neighbours.

| Method | Description |
|--------|-------------|
| `new(detector)` | Build topology from detector |
| `predict_room(room)` | `Prediction` with value, confidence, method |
| `predict_all()` | Predictions for every room |
| `update_accuracy(room, predicted, actual)` | Update spline prediction accuracy |

### `Prediction`

```rust
pub struct Prediction {
    pub room_id: String,
    pub predicted_value: f64,
    pub confidence: f64,
    pub based_on: Vec<String>,  // rooms used
    pub method: String,          // "weighted_correlation" or "fallback_mean"
}
```

## How It Works

### Data Flow

1. **Observe** — Push `(timestamp, value)` tuples into the detector per room.
2. **Correlate** — Compute pairwise Pearson *r* using the overlapping tail of each pair's time series.
3. **Classify** — Threshold the coefficient and check autocorrelation to assign a `SplineType`.
4. **Topology** — Assemble splines into a graph; compute clusters, centrality, bridges.
5. **Predict** — For each room, take a confidence-weighted average of its neighbours' latest values.

### Prediction Strategy

If a room has correlated neighbours, its predicted value is:

$$
\hat{v}_i = \frac{\sum_j |r_{ij}| \cdot c_{ij} \cdot v_j^{\text{last}}}{\sum_j |r_{ij}| \cdot c_{ij}}
$$

where $r_{ij}$ is the Pearson coefficient and $c_{ij}$ is the confidence. If no neighbours exist, it falls back to the room's own historical mean.

## The Math

### Pearson Correlation Coefficient

For two time series $X$ and $Y$ of length $n$:

$$
r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

Properties verified by the test suite:
- $r(X, X) = 1$ (self-correlation is perfect)
- $r(X, Y) = r(Y, X)$ (symmetry)
- $r(X, -X) = -1$ (anti-correlation)

### Autocorrelation

At lag $k$ for series $X$ with mean $\mu$ and variance $\sigma^2$:

$$
R(k) = \frac{\sum_{i=1}^{n-k}(x_i - \mu)(x_{i+k} - \mu)}{(n - k) \cdot \sigma^2}
$$

$R(0) = 1$ by definition.

### Spline Classification Rules

| Condition | SplineType |
|-----------|------------|
| $r > 0.95$ | `Synergistic` |
| $0.7 < r \leq 0.95$ | `Predictive` |
| $0.3 < r \leq 0.7$ and both $R_a(1) > 0.5$, $R_b(1) > 0.5$ | `Resonant` |
| $0.3 < r \leq 0.7$ otherwise | `Causal` |
| $r < -0.7$ | `Redundant` |
| $-0.7 \leq r < -0.3$ | `Causal` |
| $\|r\| \leq 0.3$ | `Resonant` (default) |

### Graph Theory

- **Clusters** are found via union-find (disjoint-set with path compression).
- **Degree centrality** = $\frac{\deg(v)}{n - 1}$.
- **Bridge rooms** are identified by brute-force removal: a room is a bridge if removing it increases the number of connected components.

## Testing

74 tests covering all data structures, correlation math, classification rules, prediction, topology analysis, serde round-trips, and formal theorem verification.

```bash
cargo test
```

## License

MIT

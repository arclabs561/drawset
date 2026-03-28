//! T-conorm (s-norm) families for fuzzy logic and differentiable relaxations.
//!
//! A t-conorm S: [0,1]^2 -> [0,1] generalizes logical OR. It is:
//! - Commutative: S(a, b) = S(b, a)
//! - Associative: S(S(a, b), c) = S(a, S(b, c))
//! - Monotone: a1 <= a2 => S(a1, b) <= S(a2, b)
//! - Bounded: S(a, 0) = a (0 is identity)
//!
//! Different families provide different "softness" of the OR operation,
//! useful in differentiable relaxations and probabilistic reasoning.
//!
//! ## Catalog
//!
//! | Family | Formula S(a,b) | Parameter | Behavior |
//! |--------|---------------|-----------|----------|
//! | Maximum | max(a,b) | none | Hard OR |
//! | Probabilistic | a+b-ab | none | Independent events |
//! | Bounded (Lukasiewicz) | min(1, a+b) | none | Saturating sum |
//! | Einstein | (a+b)/(1+ab) | none | Smooth OR |
//! | Hamacher | (a+b-2ab)/(1-ab) | none | Aggressive |
//! | Yager | min(1, (a^p + b^p)^(1/p)) | p >= 1 | Lp norm |
//! | Frank | 1 - log_s(1+(s^(1-a)-1)(s^(1-b)-1)/(s-1)) | s > 0, s != 1 | Interpolates max<->bounded |
//! | Dombi | 1/(1+((1/a-1)^p + (1/b-1)^p)^(-1/p)) | p > 0 | Power-based |
//!
//! Based on: Petersen et al., gendr -- Generalized Differentiable Rendering.
//! The t-conorm catalog provides the relaxation families used for differentiable
//! logical operations and soft aggregation.

/// Evaluate a t-conorm S(a, b) for the given family.
pub fn tconorm(family: TConormFamily, a: f64, b: f64) -> f64 {
    let a = a.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);

    match family {
        TConormFamily::Maximum => a.max(b),
        TConormFamily::Probabilistic => a + b - a * b,
        TConormFamily::Bounded => (a + b).min(1.0),
        TConormFamily::Einstein => {
            let denom = 1.0 + a * b;
            if denom == 0.0 {
                0.0
            } else {
                (a + b) / denom
            }
        }
        TConormFamily::Hamacher => {
            let denom = 1.0 - a * b;
            if denom.abs() < 1e-15 {
                1.0 // both at 1.0
            } else {
                (a + b - 2.0 * a * b) / denom
            }
        }
        TConormFamily::Yager { p } => {
            debug_assert!(p >= 1.0, "Yager p must be >= 1");
            (a.powf(p) + b.powf(p)).powf(1.0 / p).min(1.0)
        }
        TConormFamily::Frank { s } => {
            debug_assert!(s > 0.0 && s != 1.0, "Frank s must be > 0 and != 1");
            let num = (s.powf(1.0 - a) - 1.0) * (s.powf(1.0 - b) - 1.0);
            let denom = s - 1.0;
            1.0 - (1.0 + num / denom).log(s)
        }
        TConormFamily::Dombi { p } => {
            debug_assert!(p > 0.0, "Dombi p must be > 0");
            if a <= 0.0 {
                return b;
            }
            if b <= 0.0 {
                return a;
            }
            if a >= 1.0 || b >= 1.0 {
                return 1.0;
            }
            let ta = (1.0 / a - 1.0).powf(p);
            let tb = (1.0 / b - 1.0).powf(p);
            // Harmonic-like combination
            let inner = (ta.recip() + tb.recip()).recip();
            1.0 / (1.0 + inner.powf(1.0 / p))
        }
    }
}

/// T-conorm family selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TConormFamily {
    /// S(a,b) = max(a,b). Hard OR, not differentiable at a=b.
    Maximum,
    /// S(a,b) = a+b-ab. Probabilistic sum (independent events).
    Probabilistic,
    /// S(a,b) = min(1, a+b). Lukasiewicz / bounded sum.
    Bounded,
    /// S(a,b) = (a+b)/(1+ab). Smooth, well-behaved for optimization.
    Einstein,
    /// S(a,b) = (a+b-2ab)/(1-ab). More aggressive than probabilistic.
    Hamacher,
    /// S(a,b) = min(1, (a^p + b^p)^(1/p)). Lp-norm family.
    Yager {
        /// Exponent, must be >= 1. p=1 gives Bounded, p->inf gives Maximum.
        p: f64,
    },
    /// Frank t-conorm parameterized by base s.
    /// s->0 gives Maximum, s=1 gives Probabilistic (limit), s->inf gives Bounded.
    Frank {
        /// Base parameter, must be > 0 and != 1.
        s: f64,
    },
    /// Dombi t-conorm. p->0 gives Maximum, p->inf gives Bounded.
    Dombi {
        /// Exponent, must be > 0.
        p: f64,
    },
}

/// The dual t-norm T(a,b) = 1 - S(1-a, 1-b) for a given t-conorm family.
///
/// T-norms generalize logical AND.
pub fn tnorm(family: TConormFamily, a: f64, b: f64) -> f64 {
    1.0 - tconorm(family, 1.0 - a, 1.0 - b)
}

/// Aggregate multiple values using a t-conorm (generalized multi-way OR).
///
/// Folds left-to-right using associativity: S(S(S(a, b), c), d)...
pub fn tconorm_fold(family: TConormFamily, values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0, // identity element for t-conorms
        1 => values[0].clamp(0.0, 1.0),
        _ => values
            .iter()
            .skip(1)
            .fold(values[0].clamp(0.0, 1.0), |acc, &v| tconorm(family, acc, v)),
    }
}

/// Aggregate multiple values using a t-norm (generalized multi-way AND).
pub fn tnorm_fold(family: TConormFamily, values: &[f64]) -> f64 {
    match values.len() {
        0 => 1.0, // identity element for t-norms
        1 => values[0].clamp(0.0, 1.0),
        _ => values
            .iter()
            .skip(1)
            .fold(values[0].clamp(0.0, 1.0), |acc, &v| tnorm(family, acc, v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_families() -> Vec<TConormFamily> {
        vec![
            TConormFamily::Maximum,
            TConormFamily::Probabilistic,
            TConormFamily::Bounded,
            TConormFamily::Einstein,
            TConormFamily::Hamacher,
            TConormFamily::Yager { p: 2.0 },
            TConormFamily::Frank { s: 2.0 },
            TConormFamily::Dombi { p: 2.0 },
        ]
    }

    #[test]
    fn identity_element() {
        for f in all_families() {
            let r = tconorm(f, 0.5, 0.0);
            assert!(
                (r - 0.5).abs() < 1e-6,
                "{f:?}: S(0.5, 0) = {r}, expected 0.5"
            );
        }
    }

    #[test]
    fn commutativity() {
        for f in all_families() {
            let r1 = tconorm(f, 0.3, 0.7);
            let r2 = tconorm(f, 0.7, 0.3);
            assert!(
                (r1 - r2).abs() < 1e-10,
                "{f:?}: S(0.3,0.7)={r1} != S(0.7,0.3)={r2}"
            );
        }
    }

    #[test]
    fn monotonicity() {
        for f in all_families() {
            let r1 = tconorm(f, 0.3, 0.5);
            let r2 = tconorm(f, 0.6, 0.5);
            assert!(
                r2 >= r1 - 1e-10,
                "{f:?}: monotonicity violated S(0.3,0.5)={r1} > S(0.6,0.5)={r2}"
            );
        }
    }

    #[test]
    fn bounded_output() {
        for f in all_families() {
            for &a in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                for &b in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                    let r = tconorm(f, a, b);
                    assert!(
                        r >= -1e-10 && r <= 1.0 + 1e-10,
                        "{f:?}: S({a},{b}) = {r} out of [0,1]"
                    );
                }
            }
        }
    }

    #[test]
    fn probabilistic_known_values() {
        let f = TConormFamily::Probabilistic;
        // S(0.5, 0.5) = 0.5 + 0.5 - 0.25 = 0.75
        assert!((tconorm(f, 0.5, 0.5) - 0.75).abs() < 1e-10);
        // S(1, x) = 1 for all x
        assert!((tconorm(f, 1.0, 0.3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn yager_p1_equals_bounded() {
        let yager = TConormFamily::Yager { p: 1.0 };
        let bounded = TConormFamily::Bounded;
        for &a in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            for &b in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                let ry = tconorm(yager, a, b);
                let rb = tconorm(bounded, a, b);
                assert!(
                    (ry - rb).abs() < 1e-10,
                    "Yager(p=1) != Bounded: S({a},{b}) = {ry} vs {rb}"
                );
            }
        }
    }

    #[test]
    fn tnorm_duality() {
        // T(a,b) = 1 - S(1-a, 1-b)
        // For probabilistic: T(a,b) = ab
        let f = TConormFamily::Probabilistic;
        let t = tnorm(f, 0.5, 0.6);
        assert!((t - 0.3).abs() < 1e-10, "T(0.5,0.6) = {t}, expected 0.3");
    }

    #[test]
    fn fold_empty_returns_identity() {
        let f = TConormFamily::Maximum;
        assert_eq!(tconorm_fold(f, &[]), 0.0);
        assert_eq!(tnorm_fold(f, &[]), 1.0);
    }

    #[test]
    fn fold_single_value() {
        let f = TConormFamily::Probabilistic;
        assert!((tconorm_fold(f, &[0.7]) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn fold_associativity() {
        let f = TConormFamily::Einstein;
        let abc = tconorm_fold(f, &[0.3, 0.5, 0.7]);
        let ab_c = tconorm(f, tconorm(f, 0.3, 0.5), 0.7);
        assert!(
            (abc - ab_c).abs() < 1e-10,
            "fold should match left-to-right application"
        );
    }
}

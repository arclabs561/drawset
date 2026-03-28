//! Quasi-Monte Carlo sequences.
//!
//! Low-discrepancy sequences fill the unit hypercube more uniformly than
//! pseudorandom points.  For numerical integration, QMC achieves
//! $O((\log n)^d / n)$ error vs $O(1/\sqrt{n})$ for plain MC.
//!
//! ## Sequences
//!
//! - **Halton**: radical-inverse construction using successive primes as bases.
//!   Simple, no precomputation, but quality degrades above ~20 dimensions
//!   due to correlation between coordinates.
//! - **Sobol**: direction-number construction with Gray-code enumeration.
//!   Better high-dimensional uniformity than Halton.  Direction numbers from
//!   Joe & Kuo (2010) support up to dimension 1111.
//! - **Owen-scrambled Sobol**: random digit scrambling that preserves the
//!   low-discrepancy property while breaking systematic patterns.  Enables
//!   unbiased variance estimation from a single randomized QMC sequence.
//!
//! ## References
//!
//! - Halton (1960): "On the efficiency of certain quasi-random sequences of points
//!   in evaluating multi-dimensional integrals."
//! - Sobol (1967): "Distribution of points in a cube and approximate evaluation
//!   of integrals."
//! - Joe & Kuo (2010): "Constructing Sobol sequences with better two-dimensional
//!   projections."
//! - Owen (1995): "Randomly permuted (t,m,s)-nets and (t,s)-sequences."
//! - Martinez & Williams (2026): "QMC Methods Enable Extremely Low-Dimensional
//!   Deep Generative Models."

// ---------------------------------------------------------------------------
// Halton sequence
// ---------------------------------------------------------------------------

/// First 20 primes, used as bases for Halton coordinates.
const PRIMES: [u64; 20] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
];

/// Radical inverse of `index` in the given `base`.
///
/// Maps a non-negative integer to `[0, 1)` by reflecting its digits in `base`
/// around the decimal point.  For example, `radical_inverse(5, 2)` = 0.101 in
/// binary = 0.625.
fn radical_inverse(mut index: u64, base: u64) -> f64 {
    let mut result = 0.0_f64;
    let mut denom = 1.0_f64;
    while index > 0 {
        denom *= base as f64;
        result += (index % base) as f64 / denom;
        index /= base;
    }
    result
}

/// Compute a single Halton point in `[0, 1)^d`.
///
/// Coordinate `j` uses the `(j+1)`-th prime as its radical-inverse base.
///
/// # Panics
///
/// Panics if `d` exceeds 20 (the number of precomputed prime bases).
///
/// # Examples
///
/// ```
/// let p = kuji::qmc::halton_point(1, 3);
/// assert!((p[0] - 0.5).abs() < 1e-12);
/// assert!((p[1] - 1.0 / 3.0).abs() < 1e-12);
/// assert!((p[2] - 0.2).abs() < 1e-12);
/// ```
pub fn halton_point(index: usize, d: usize) -> Vec<f64> {
    assert!(
        d <= PRIMES.len(),
        "Halton supports at most {} dimensions",
        PRIMES.len()
    );
    (0..d)
        .map(|j| radical_inverse(index as u64, PRIMES[j]))
        .collect()
}

/// Generate `n` Halton points in `[0, 1)^d`, starting from `index = 1`.
///
/// Index 0 is skipped because it maps to the origin in every dimension.
///
/// # Panics
///
/// Panics if `d` exceeds 20.
///
/// # Examples
///
/// ```
/// let pts = kuji::qmc::halton_sequence(4, 2);
/// assert_eq!(pts.len(), 4);
/// // base-2 first four: 0.5, 0.25, 0.75, 0.125
/// assert!((pts[0][0] - 0.5).abs() < 1e-12);
/// assert!((pts[1][0] - 0.25).abs() < 1e-12);
/// assert!((pts[2][0] - 0.75).abs() < 1e-12);
/// assert!((pts[3][0] - 0.125).abs() < 1e-12);
/// ```
pub fn halton_sequence(n: usize, d: usize) -> Vec<Vec<f64>> {
    (1..=n).map(|i| halton_point(i, d)).collect()
}

// ---------------------------------------------------------------------------
// Sobol sequence
// ---------------------------------------------------------------------------

/// Number of bits used in the Sobol generator (matches f64 mantissa capacity).
const SOBOL_BITS: usize = 52;

/// Direction numbers for dimensions 2..=8 (dimension 1 is the Van der Corput
/// sequence and needs no table).
///
/// Each row stores `(degree, [coefficients of primitive polynomial], [initial direction numbers])`.
/// Source: Joe & Kuo (2010), truncated to the first 7 extra dimensions.
///
/// Format: `(s, a, [m_1, m_2, ..., m_s])` where `s` = degree of the primitive
/// polynomial over GF(2), `a` = binary representation of the polynomial
/// coefficients (excluding leading and constant terms).
const JOE_KUO_D2_D8: [(u32, u32, &[u32]); 7] = [
    // dim 2: s=1, poly = x+1, a=0, m=[1]
    (1, 0, &[1]),
    // dim 3: s=2, poly = x^2+x+1, a=1, m=[1,1]
    (2, 1, &[1, 1]),
    // dim 4: s=3, poly = x^3+x+1, a=1, m=[1,1,1]
    (3, 1, &[1, 1, 1]),
    // dim 5: s=3, poly = x^3+x^2+1, a=2, m=[1,3,1]
    (3, 2, &[1, 3, 1]),
    // dim 6: s=4, poly = x^4+x+1, a=1, m=[1,1,3,3]
    (4, 1, &[1, 1, 3, 3]),
    // dim 7: s=4, poly = x^4+x^3+1, a=4, m=[1,3,5,13]
    (4, 4, &[1, 3, 5, 13]),
    // dim 8: s=5, poly = x^5+x^2+1, a=2, m=[1,1,5,5,17]
    (5, 2, &[1, 1, 5, 5, 17]),
];

/// Build the full direction-number table `v[j][i]` for dimension index `j`
/// (0-based, so j=0 is dimension 1 = Van der Corput).
fn build_direction_numbers(dim: usize) -> Vec<[u64; SOBOL_BITS]> {
    let mut v = Vec::with_capacity(dim);

    // Dimension 1: Van der Corput in base 2.
    {
        let mut row = [0u64; SOBOL_BITS];
        for (i, slot) in row.iter_mut().enumerate() {
            *slot = 1u64 << (SOBOL_BITS - 1 - i);
        }
        v.push(row);
    }

    // Dimensions 2..=dim.
    for &(s, a, m_init) in JOE_KUO_D2_D8.iter().take(dim.saturating_sub(1)) {
        let s = s as usize;
        let mut row = [0u64; SOBOL_BITS];

        // Initial direction numbers, left-shifted.
        for i in 0..s {
            row[i] = (m_init[i] as u64) << (SOBOL_BITS - 1 - i);
        }

        // Recurrence: v_i = (v_{i-s} >> s) XOR v_{i-s}
        //             XOR sum_{k=1}^{s-1} bit_k(a) * v_{i-k}
        //             where bit_k means the k-th bit from MSB of `a`.
        for i in s..SOBOL_BITS {
            let mut val = row[i - s] >> s;
            val ^= row[i - s];
            for k in 1..s {
                // bit k of `a` (1-indexed from MSB of the s-1 bit representation).
                if (a >> (s - 1 - k)) & 1 == 1 {
                    val ^= row[i - k];
                }
            }
            row[i] = val;
        }
        v.push(row);
    }

    v
}

/// Position of the rightmost zero bit in `n` (0-indexed).
///
/// For Gray-code Sobol generation, this determines which direction number
/// to XOR with the current point.
fn rightmost_zero(n: u64) -> usize {
    // Flip bits -> rightmost zero becomes rightmost one -> trailing_zeros.
    (!n).trailing_zeros() as usize
}

/// Incremental Sobol sequence generator using Gray-code enumeration.
///
/// Each call to [`next`](SobolGenerator::next) returns the next point and
/// advances the internal counter.  The generator supports up to 8 dimensions
/// (1 Van der Corput + 7 from the Joe-Kuo table).
///
/// # Examples
///
/// ```
/// let mut gen = kuji::qmc::SobolGenerator::new(2);
/// let p0 = gen.next(); // index 0 -> origin (often skipped)
/// let p1 = gen.next(); // index 1
/// assert!((p1[0] - 0.5).abs() < 1e-12);
/// assert!((p1[1] - 0.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct SobolGenerator {
    dim: usize,
    index: u64,
    x: Vec<u64>,
    v: Vec<[u64; SOBOL_BITS]>,
}

impl SobolGenerator {
    /// Create a new Sobol generator for `dim` dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is 0 or exceeds 8.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dimension must be >= 1");
        assert!(
            dim <= 8,
            "Sobol supports at most 8 dimensions (embedded table)"
        );
        let v = build_direction_numbers(dim);
        Self {
            dim,
            index: 0,
            x: vec![0u64; dim],
            v,
        }
    }

    /// Return the next Sobol point in `[0, 1)^d` and advance the counter.
    #[allow(clippy::should_implement_trait)] // not a standard Iterator (returns Vec, stateful)
    pub fn next(&mut self) -> Vec<f64> {
        if self.index == 0 {
            self.index = 1;
            // First point is the origin.
            return vec![0.0; self.dim];
        }

        let c = rightmost_zero(self.index - 1);
        for j in 0..self.dim {
            self.x[j] ^= self.v[j][c.min(SOBOL_BITS - 1)];
        }
        self.index += 1;

        let scale = (1u64 << SOBOL_BITS) as f64;
        self.x.iter().map(|&xi| xi as f64 / scale).collect()
    }

    /// Skip `n` points without allocating output vectors.
    pub fn skip(&mut self, n: u64) {
        for _ in 0..n {
            if self.index == 0 {
                self.index = 1;
                continue;
            }
            let c = rightmost_zero(self.index - 1);
            for j in 0..self.dim {
                self.x[j] ^= self.v[j][c.min(SOBOL_BITS - 1)];
            }
            self.index += 1;
        }
    }

    /// Current index (number of points already generated, including the origin).
    pub fn index(&self) -> u64 {
        self.index
    }
}

/// Generate `n` Sobol points in `[0, 1)^d`, starting from index 1
/// (the origin at index 0 is skipped).
///
/// # Panics
///
/// Panics if `d` is 0 or exceeds 8.
///
/// # Examples
///
/// ```
/// let pts = kuji::qmc::sobol_sequence(4, 1);
/// // 1D Sobol (Gray-code order): 0.5, 0.75, 0.25, 0.375
/// assert!((pts[0][0] - 0.5).abs() < 1e-12);
/// assert!((pts[1][0] - 0.75).abs() < 1e-12);
/// assert!((pts[2][0] - 0.25).abs() < 1e-12);
/// assert!((pts[3][0] - 0.375).abs() < 1e-12);
/// ```
pub fn sobol_sequence(n: usize, d: usize) -> Vec<Vec<f64>> {
    let mut gen = SobolGenerator::new(d);
    gen.skip(1); // skip origin
    (0..n).map(|_| gen.next()).collect()
}

// ---------------------------------------------------------------------------
// Owen-scrambled Sobol
// ---------------------------------------------------------------------------

/// Simple hash-based Owen scrambling.
///
/// Applies a bitwise random permutation seeded by `(seed, dimension)` that
/// preserves the stratification properties of the Sobol sequence.  This is
/// a simplified version of Owen's nested uniform scrambling (1995) using
/// a hash function instead of a full tree permutation.
fn owen_scramble(mut x: u64, dim: u64, seed: u64) -> u64 {
    // Use a simple xorshift-multiply hash conditioned on the prefix of bits
    // already fixed.  Each bit i is scrambled based on all higher bits and
    // the seed, matching the structure of Owen's tree-based scramble.
    for i in 0..SOBOL_BITS {
        let bit_mask = 1u64 << (SOBOL_BITS - 1 - i);
        // Hash the prefix (bits above position i) combined with seed and dim.
        let prefix = x >> (SOBOL_BITS - i);
        let mut h = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(dim)
            .wrapping_mul(0x517C_C1B7_2722_0A95)
            .wrapping_add(prefix)
            .wrapping_mul(0x6C62_272E_07BB_0142);
        h ^= h >> 32;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 31;
        // Flip bit i with probability ~0.5 based on hash.
        if h & 1 == 1 {
            x ^= bit_mask;
        }
    }
    x
}

/// Generate `n` Owen-scrambled Sobol points in `[0, 1)^d`.
///
/// The scrambling preserves low-discrepancy while randomizing the sequence,
/// enabling unbiased variance estimation from a single QMC run.
///
/// # Panics
///
/// Panics if `d` is 0 or exceeds 8.
///
/// # Examples
///
/// ```
/// let pts = kuji::qmc::sobol_scrambled(100, 2, 42);
/// assert_eq!(pts.len(), 100);
/// for p in &pts {
///     assert!(p[0] >= 0.0 && p[0] < 1.0);
///     assert!(p[1] >= 0.0 && p[1] < 1.0);
/// }
/// ```
pub fn sobol_scrambled(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut gen = SobolGenerator::new(d);
    gen.skip(1); // skip origin
    let scale = (1u64 << SOBOL_BITS) as f64;
    (0..n)
        .map(|_| {
            // Generate unscrambled integer point.
            if gen.index == 0 {
                gen.index = 1;
            } else {
                let c = rightmost_zero(gen.index - 1);
                for j in 0..gen.dim {
                    gen.x[j] ^= gen.v[j][c.min(SOBOL_BITS - 1)];
                }
                gen.index += 1;
            }
            // Scramble each coordinate independently.
            (0..d)
                .map(|j| {
                    let scrambled = owen_scramble(gen.x[j], j as u64, seed);
                    scrambled as f64 / scale
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------- Halton -------

    #[test]
    fn halton_base2_known_values() {
        // Base-2 radical inverse: 1->0.5, 2->0.25, 3->0.75, 4->0.125
        let pts = halton_sequence(4, 1);
        let expected = [0.5, 0.25, 0.75, 0.125];
        for (p, &e) in pts.iter().zip(&expected) {
            assert!((p[0] - e).abs() < 1e-12, "got {} expected {}", p[0], e);
        }
    }

    #[test]
    fn halton_base3_known_values() {
        // Base-3: 1->1/3, 2->2/3, 3->1/9, 4->4/9
        let pts = halton_sequence(4, 2);
        let expected_d1 = [1.0 / 3.0, 2.0 / 3.0, 1.0 / 9.0, 4.0 / 9.0];
        for (p, &e) in pts.iter().zip(&expected_d1) {
            assert!((p[1] - e).abs() < 1e-12, "got {} expected {}", p[1], e);
        }
    }

    #[test]
    fn halton_in_unit_cube() {
        let pts = halton_sequence(1000, 5);
        for p in &pts {
            assert_eq!(p.len(), 5);
            for &x in p {
                assert!(x >= 0.0 && x < 1.0, "out of range: {x}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "at most 20 dimensions")]
    fn halton_rejects_high_dim() {
        halton_point(1, 21);
    }

    // ------- Sobol -------

    #[test]
    fn sobol_1d_known_values() {
        // 1D Sobol with Gray-code enumeration: 0.5, 0.75, 0.25, 0.375, ...
        // (differs from natural-order Van der Corput because Gray-code
        // XORs direction numbers in a different order).
        let pts = sobol_sequence(4, 1);
        let expected = [0.5, 0.75, 0.25, 0.375];
        for (p, &e) in pts.iter().zip(&expected) {
            assert!((p[0] - e).abs() < 1e-12, "got {} expected {}", p[0], e);
        }
    }

    #[test]
    fn sobol_2d_first_point() {
        let pts = sobol_sequence(1, 2);
        // First point (index 1): both dims should be 0.5.
        assert!((pts[0][0] - 0.5).abs() < 1e-12);
        assert!((pts[0][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sobol_in_unit_cube() {
        let pts = sobol_sequence(1024, 4);
        for p in &pts {
            assert_eq!(p.len(), 4);
            for &x in p {
                assert!(x >= 0.0 && x < 1.0, "out of range: {x}");
            }
        }
    }

    #[test]
    fn sobol_generator_skip_consistent() {
        // Generating 10 points should equal skipping 5 then generating 5,
        // prepended with the first 5 from a fresh generator.
        let mut gen1 = SobolGenerator::new(3);
        let all: Vec<_> = (0..10).map(|_| gen1.next()).collect();

        let mut gen2 = SobolGenerator::new(3);
        gen2.skip(5);
        let last5: Vec<_> = (0..5).map(|_| gen2.next()).collect();

        for (a, b) in all[5..].iter().zip(&last5) {
            assert_eq!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "at most 8 dimensions")]
    fn sobol_rejects_high_dim() {
        SobolGenerator::new(9);
    }

    #[test]
    #[should_panic(expected = "dimension must be >= 1")]
    fn sobol_rejects_zero_dim() {
        SobolGenerator::new(0);
    }

    // ------- Owen-scrambled Sobol -------

    #[test]
    fn scrambled_in_unit_cube() {
        let pts = sobol_scrambled(512, 3, 12345);
        for p in &pts {
            assert_eq!(p.len(), 3);
            for &x in p {
                assert!(x >= 0.0 && x < 1.0, "out of range: {x}");
            }
        }
    }

    #[test]
    fn scrambled_different_seeds_differ() {
        let a = sobol_scrambled(16, 2, 1);
        let b = sobol_scrambled(16, 2, 2);
        // At least some points should differ.
        let diffs = a.iter().zip(&b).filter(|(pa, pb)| pa != pb).count();
        assert!(
            diffs > 0,
            "different seeds should produce different sequences"
        );
    }

    #[test]
    fn scrambled_same_seed_reproducible() {
        let a = sobol_scrambled(32, 2, 42);
        let b = sobol_scrambled(32, 2, 42);
        assert_eq!(a, b);
    }

    // ------- Integration test: QMC vs MC convergence -------

    #[test]
    fn qmc_integrates_x_squared() {
        // Integral of x^2 on [0,1] = 1/3.
        let exact = 1.0 / 3.0;
        let n = 4096;

        // QMC estimate (Sobol).
        let pts = sobol_sequence(n, 1);
        let qmc_est: f64 = pts.iter().map(|p| p[0] * p[0]).sum::<f64>() / n as f64;
        let qmc_err = (qmc_est - exact).abs();

        // MC estimate (Halton as a simpler stand-in -- still QMC but independent).
        // For a true comparison, use pseudorandom, but we just verify QMC is close.
        assert!(qmc_err < 5e-4, "QMC integration error too large: {qmc_err}");
    }

    #[test]
    fn halton_integrates_x_squared() {
        let exact = 1.0 / 3.0;
        let n = 4096;
        let pts = halton_sequence(n, 1);
        let est: f64 = pts.iter().map(|p| p[0] * p[0]).sum::<f64>() / n as f64;
        let err = (est - exact).abs();
        assert!(err < 5e-4, "Halton integration error too large: {err}");
    }

    // ------- Helpers -------

    #[test]
    fn rightmost_zero_cases() {
        assert_eq!(rightmost_zero(0b0), 0); // ...0 -> bit 0
        assert_eq!(rightmost_zero(0b1), 1); // ...1 -> bit 1
        assert_eq!(rightmost_zero(0b11), 2);
        assert_eq!(rightmost_zero(0b101), 1);
        assert_eq!(rightmost_zero(0b111), 3);
    }

    #[test]
    fn radical_inverse_base2() {
        assert!((radical_inverse(1, 2) - 0.5).abs() < 1e-15);
        assert!((radical_inverse(2, 2) - 0.25).abs() < 1e-15);
        assert!((radical_inverse(3, 2) - 0.75).abs() < 1e-15);
        assert!((radical_inverse(5, 2) - 0.625).abs() < 1e-15);
    }

    #[test]
    fn radical_inverse_base3() {
        // 1 in base 3 = "1" -> 0.1 = 1/3
        assert!((radical_inverse(1, 3) - 1.0 / 3.0).abs() < 1e-15);
        // 2 in base 3 = "2" -> 0.2 = 2/3
        assert!((radical_inverse(2, 3) - 2.0 / 3.0).abs() < 1e-15);
        // 3 in base 3 = "10" -> 0.01 = 1/9
        assert!((radical_inverse(3, 3) - 1.0 / 9.0).abs() < 1e-15);
    }
}

//! AVX2 + FMA vector math. `exp`, `log` and `tanh` have no vector form in libm, so the compiler cannot
//! vectorize them; these are Cephes-style polynomial versions accurate to a few ulps in f32.
//! Everything here is `x86_64` only and gated on [`has_avx2`]; other targets use scalar code.

/// True when the CPU has AVX2 and FMA (cached by the standard library after the first call).
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp256(x: __m256) -> __m256 {
        let nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x);
        let too_big = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(88.722_84));
        // Clamp to the range where the result is representable (denormals included at the bottom).
        let xc = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.722_84)), _mm256_set1_ps(-103.972_08));
        let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(xc, _mm256_set1_ps(std::f32::consts::LOG2_E)));
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), xc);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
        let mut y = _mm256_set1_ps(1.987_569_1e-4);
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.398_199_9e-3));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.333_452e-3));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.166_579_6e-2));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.666_666_5e-1));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.000_000_1e-1));
        y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), r);
        y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
        // 2^n as two factors, so n up to 128 (or down to -150) never overflows the exponent field.
        let ni = _mm256_cvtps_epi32(n);
        let n1 = _mm256_srai_epi32::<1>(ni);
        let n2 = _mm256_sub_epi32(ni, n1);
        let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, _mm256_set1_epi32(127))));
        let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n2, _mm256_set1_epi32(127))));
        let res = _mm256_mul_ps(_mm256_mul_ps(y, s1), s2);
        let res = _mm256_blendv_ps(res, _mm256_set1_ps(f32::INFINITY), too_big);
        _mm256_blendv_ps(res, x, nan)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn log256(x: __m256) -> __m256 {
        let zero = _mm256_setzero_ps();
        let is_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x);
        let neg = _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero);
        let is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, zero);
        let is_inf = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_set1_ps(f32::INFINITY));
        // Denormals: scale up by 2^23 and correct the exponent afterwards.
        let denorm = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x0080_0000)));
        let xs = _mm256_blendv_ps(x, _mm256_mul_ps(x, _mm256_set1_ps(8_388_608.0)), denorm);
        let adj = _mm256_and_ps(denorm, _mm256_set1_ps(23.0));

        let bits = _mm256_castps_si256(xs);
        let emm = _mm256_srli_epi32::<23>(bits);
        let m = _mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
            _mm256_set1_epi32(0x3f00_0000),
        ));
        let one = _mm256_set1_ps(1.0);
        let mut e = _mm256_cvtepi32_ps(_mm256_sub_epi32(emm, _mm256_set1_epi32(0x7f)));
        e = _mm256_sub_ps(_mm256_add_ps(e, one), adj);
        let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(m, _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2));
        let tmp = _mm256_and_ps(m, mask);
        let mut x1 = _mm256_sub_ps(m, one);
        e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
        x1 = _mm256_add_ps(x1, tmp);
        let z = _mm256_mul_ps(x1, x1);
        let mut y = _mm256_set1_ps(7.037_683_6e-2);
        y = _mm256_fmadd_ps(y, x1, _mm256_set1_ps(-1.151_461e-1));
        y = _mm256_fmadd_ps(y, x1, _mm256_set1_ps(1.167_699_9e-1));
        y = _mm256_fmadd_ps(y, x1, _mm256_set1_ps(-1.242_014_1e-1));
        y = _mm256_fmadd_ps(y, x1, _mm256_set1_ps(1.424_932_3e-1));
        y = _mm256_fmadd_ps(y, x1, _mm256_set1_ps(-1.666_805_8e-1));
        y = _mm256_fmadd_ps(y, x1, _mm256_set1_ps(2.000_071_5e-1));
        y = _mm256_fmadd_ps(y, x1, _mm256_set1_ps(-2.499_999_4e-1));
        y = _mm256_fmadd_ps(y, x1, _mm256_set1_ps(3.333_333_2e-1));
        y = _mm256_mul_ps(_mm256_mul_ps(y, x1), z);
        y = _mm256_fmadd_ps(e, _mm256_set1_ps(-2.121_944_4e-4), y);
        y = _mm256_fnmadd_ps(z, _mm256_set1_ps(0.5), y);
        let mut res = _mm256_add_ps(x1, y);
        res = _mm256_fmadd_ps(e, _mm256_set1_ps(0.693_359_4), res);

        res = _mm256_blendv_ps(res, _mm256_set1_ps(f32::INFINITY), is_inf);
        res = _mm256_blendv_ps(res, _mm256_set1_ps(f32::NEG_INFINITY), is_zero);
        res = _mm256_blendv_ps(res, _mm256_set1_ps(f32::NAN), neg);
        _mm256_blendv_ps(res, x, is_nan)
    }

    /// Rational approximation of tanh (the same form Eigen uses): good to a few ulps for |x| < 7.9,
    /// saturating beyond.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tanh256(x: __m256) -> __m256 {
        let nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x);
        let lim = _mm256_set1_ps(7.905_311);
        let xc = _mm256_max_ps(_mm256_min_ps(x, lim), _mm256_sub_ps(_mm256_setzero_ps(), lim));
        let x2 = _mm256_mul_ps(xc, xc);
        let mut p = _mm256_set1_ps(-2.760_768_5e-16);
        for c in [2.000_187_9e-13f32, -8.604_672e-11, 5.122_297e-8, 1.485_722_4e-5, 6.372_619_3e-4, 4.893_524_5e-3] {
            p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(c));
        }
        p = _mm256_mul_ps(p, xc);
        let mut q = _mm256_set1_ps(1.198_258_4e-6);
        for c in [1.185_347e-4f32, 2.268_434_6e-3, 4.893_525e-3] {
            q = _mm256_fmadd_ps(q, x2, _mm256_set1_ps(c));
        }
        _mm256_blendv_ps(_mm256_div_ps(p, q), x, nan)
    }
}

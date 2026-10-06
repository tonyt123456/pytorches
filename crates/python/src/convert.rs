//! Element types PyTorches can read from foreign buffers (DLPack, safetensors) and widen to `f32`.

/// A foreign element type. Everything is converted to `f32` on the way in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F32,
    F64,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl Elem {
    pub fn size(self) -> usize {
        match self {
            Elem::F32 | Elem::I32 | Elem::U32 => 4,
            Elem::F64 | Elem::I64 | Elem::U64 => 8,
            Elem::F16 | Elem::BF16 | Elem::I16 | Elem::U16 => 2,
            Elem::I8 | Elem::U8 | Elem::Bool => 1,
        }
    }

    /// safetensors dtype names.
    pub fn from_safetensors(name: &str) -> Option<Elem> {
        Some(match name {
            "F32" => Elem::F32,
            "F64" => Elem::F64,
            "F16" => Elem::F16,
            "BF16" => Elem::BF16,
            "I8" => Elem::I8,
            "I16" => Elem::I16,
            "I32" => Elem::I32,
            "I64" => Elem::I64,
            "U8" => Elem::U8,
            "U16" => Elem::U16,
            "U32" => Elem::U32,
            "U64" => Elem::U64,
            "BOOL" => Elem::Bool,
            _ => return None,
        })
    }

    /// DLPack `(code, bits, lanes)`: code 0 = int, 1 = uint, 2 = float, 4 = bfloat, 6 = bool.
    pub fn from_dlpack(code: u8, bits: u8, lanes: u16) -> Option<Elem> {
        if lanes != 1 {
            return None;
        }
        Some(match (code, bits) {
            (2, 32) => Elem::F32,
            (2, 64) => Elem::F64,
            (2, 16) => Elem::F16,
            (4, 16) => Elem::BF16,
            (0, 8) => Elem::I8,
            (0, 16) => Elem::I16,
            (0, 32) => Elem::I32,
            (0, 64) => Elem::I64,
            (1, 8) => Elem::U8,
            (1, 16) => Elem::U16,
            (1, 32) => Elem::U32,
            (1, 64) => Elem::U64,
            (6, 8) => Elem::Bool,
            _ => return None,
        })
    }

    /// Reads one element at `ptr` (no alignment requirement) and widens it to `f32`.
    ///
    /// # Safety
    /// `ptr` must be valid for `self.size()` bytes.
    pub unsafe fn read(self, ptr: *const u8) -> f32 {
        unsafe {
            match self {
                Elem::F32 => (ptr as *const f32).read_unaligned(),
                Elem::F64 => (ptr as *const f64).read_unaligned() as f32,
                Elem::F16 => f16_to_f32((ptr as *const u16).read_unaligned()),
                Elem::BF16 => f32::from_bits((((ptr as *const u16).read_unaligned()) as u32) << 16),
                Elem::I8 => (ptr as *const i8).read_unaligned() as f32,
                Elem::I16 => (ptr as *const i16).read_unaligned() as f32,
                Elem::I32 => (ptr as *const i32).read_unaligned() as f32,
                Elem::I64 => (ptr as *const i64).read_unaligned() as f32,
                Elem::U8 => ptr.read() as f32,
                Elem::U16 => (ptr as *const u16).read_unaligned() as f32,
                Elem::U32 => (ptr as *const u32).read_unaligned() as f32,
                Elem::U64 => (ptr as *const u64).read_unaligned() as f32,
                Elem::Bool => (ptr.read() != 0) as u8 as f32,
            }
        }
    }
}

/// IEEE 754 binary16 to binary32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, f) => {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = f;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, f) => (sign << 31) | 0x7f80_0000 | (f << 13),
        (e, f) => (sign << 31) | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// Row-major strides (in elements) for `shape`.
pub fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc = 1i64;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i].max(1) as i64;
    }
    strides
}

/// True if `strides` describe `shape` laid out contiguously (size-1 dims may have any stride).
pub fn is_contiguous(shape: &[usize], strides: &[i64]) -> bool {
    let mut expect = 1i64;
    for i in (0..shape.len()).rev() {
        if shape[i] != 1 && strides[i] != expect {
            return false;
        }
        expect *= shape[i].max(1) as i64;
    }
    true
}

/// Gathers `shape`-many elements of type `elem` from `base` through element `strides`,
/// widening to `f32`. Output is row-major.
///
/// # Safety
/// Every addressed element must be valid to read.
pub unsafe fn gather_f32(base: *const u8, elem: Elem, shape: &[usize], strides: &[i64]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; shape.len()];
    let mut off = 0i64;
    for _ in 0..n {
        out.push(unsafe { elem.read(base.offset(off as isize * elem.size() as isize)) });
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            off += strides[d];
            if idx[d] < shape[d] {
                break;
            }
            off -= strides[d] * shape[d] as i64;
            idx[d] = 0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24)); // smallest subnormal
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
    }

    #[test]
    fn gather_transposed_view() {
        // 2x3 row-major data read as its 3x2 transpose (strides [1, 3]).
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = unsafe { gather_f32(data.as_ptr() as *const u8, Elem::F32, &[3, 2], &[1, 3]) };
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn contiguity() {
        assert!(is_contiguous(&[2, 3], &[3, 1]));
        assert!(!is_contiguous(&[3, 2], &[1, 3]));
        assert!(is_contiguous(&[1, 4], &[99, 1])); // size-1 dim stride is irrelevant
    }
}

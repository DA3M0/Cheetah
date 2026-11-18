//! Raw-byte views over typed weight payloads.
//!
//! The registration's operator tables pass weight bytes uniformly (one
//! signature per operator for every precision); these helpers restore the
//! typed view inside each format's implementation without copying. Callers
//! only ever pass slices produced by `WeightData::as_bytes` for the same
//! format, so the alignment and length invariants hold by construction —
//! the checks below turn a violated invariant into an error instead of
//! undefined behavior.

/// Restore an f32 view over raw weight bytes (4-byte little-endian values).
pub(crate) fn f32_view(bytes: &[u8]) -> Result<&[f32], String> {
    if bytes.len() % 4 != 0 || bytes.as_ptr() as usize % 4 != 0 {
        return Err("F32 weight bytes are not a 4-aligned f32 payload.".to_string());
    }
    // SAFETY: the bytes come from a `WeightData::F32` payload (`Arc<[f32]>`),
    // so the address is 4-aligned and the length a multiple of four; the
    // f32 view covers exactly the same allocation.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) })
}

/// Restore a 16-bit view over raw weight bytes (2-byte little-endian
/// values; the f16 and bf16 pipelines both consume `u16` bit patterns).
pub(crate) fn u16_view(bytes: &[u8]) -> Result<&[u16], String> {
    if bytes.len() % 2 != 0 || bytes.as_ptr() as usize % 2 != 0 {
        return Err("16-bit weight bytes are not a 2-aligned payload.".to_string());
    }
    // SAFETY: the bytes come from a `WeightData::F16`/`BF16` payload
    // (`Arc<[u16]>`), so the address is 2-aligned and the length a multiple
    // of two; the u16 view covers exactly the same allocation.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), bytes.len() / 2) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_round_trip_values() {
        let values: [f32; 3] = [1.5, -2.25, 0.0];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
        assert_eq!(f32_view(bytes).unwrap(), &values);

        let halves: [u16; 3] = [0x3fc0, 0x4000, 0x0000]; // bf16 bits of 1.5 / 2.0 / 0.0
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(halves.as_ptr().cast::<u8>(), halves.len() * 2) };
        assert_eq!(u16_view(bytes).unwrap(), &halves);
    }

    #[test]
    fn views_reject_bad_lengths() {
        assert!(f32_view(&[0u8; 3]).is_err());
        assert!(u16_view(&[0u8; 1]).is_err());
    }
}

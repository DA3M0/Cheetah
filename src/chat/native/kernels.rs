//! Compatibility alias for the native compute kernels.
//!
//! The kernel implementations moved to [`super::ops`] (the OpsToolbox module)
//! when the per-model forward assemblies moved into the registration files;
//! the function names (`gemv`, `rmsnorm`, `rope_in_place`, `gemv_q8`, ...)
//! are unchanged, and the precision operator registry keeps reaching them
//! through this path (`crate::chat::native::kernels`).

pub use super::ops::*;

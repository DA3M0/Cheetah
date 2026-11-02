//! Safe Rust adapters for C APIs declared by the root-level C headers.

use std::ffi::CString;
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Mutex;

static INIT_LOCK: Mutex<()> = Mutex::new(());

unsafe extern "C" {
    fn cheetah_init_model(src_dir: *const c_char, dst_dir: *const c_char) -> i32;
}

fn path_to_cstring(path: &Path) -> Result<CString, String> {
    let path = path
        .to_str()
        .ok_or_else(|| format!("Path is not valid UTF-8: {}", path.display()))?;
    CString::new(path).map_err(|_| "Path contains an interior NUL byte.".to_string())
}

/// Run the C static model-preparation stage and write its result to `dst_dir`.
pub fn init_model(src_dir: &Path, dst_dir: &Path) -> Result<(), String> {
    let src_dir = path_to_cstring(src_dir)?;
    let dst_dir = path_to_cstring(dst_dir)?;
    let _guard = INIT_LOCK
        .lock()
        .map_err(|_| "Native model initializer lock is poisoned.".to_string())?;
    let result = unsafe { cheetah_init_model(src_dir.as_ptr(), dst_dir.as_ptr()) };
    if result == 0 {
        Ok(())
    } else {
        Err("Cheetah model preparation failed; see the native diagnostic above.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializer_ffi_reports_native_failure() {
        let error = init_model(
            Path::new("/cheetah-test-path-that-does-not-exist"),
            Path::new("/tmp/cheetah-native-ffi-test"),
        )
        .unwrap_err();
        assert!(error.contains("native diagnostic"));
    }
}

#[cfg(feature = "vulkan")]
pub mod vulkan {
    use std::os::raw::c_int;
    use std::sync::Mutex;

    static VULKAN_LOCK: Mutex<()> = Mutex::new(());

    unsafe extern "C" {
        fn init_vulkan_device() -> c_int;
        fn shutdown_vulkan_device();
        fn vk_matmul_checked(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
    }

    /// Initialize the C Vulkan backend and its mapped memory pool.
    pub fn initialize() -> Result<(), String> {
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let result = unsafe { init_vulkan_device() };
        if result == 0 {
            Ok(())
        } else {
            Err("Vulkan backend initialization failed; see the native diagnostic above.".into())
        }
    }

    /// Release the Vulkan backend. No allocations or submitted work may remain.
    pub fn shutdown() -> Result<(), String> {
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        unsafe { shutdown_vulkan_device() }
        Ok(())
    }

    /// Compute row-major `C[M,N] = A[M,K] * B[K,N]` using the Vulkan backend.
    pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, String> {
        if m == 0 || k == 0 || n == 0 {
            return Err("Vulkan matrix dimensions must be positive.".into());
        }
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| "A matrix element count overflows usize.".to_string())?;
        let expected_b = k
            .checked_mul(n)
            .ok_or_else(|| "B matrix element count overflows usize.".to_string())?;
        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| "C matrix element count overflows usize.".to_string())?;
        if a.len() != expected_a || b.len() != expected_b {
            return Err(format!(
                "Vulkan matmul expected A/B lengths {expected_a}/{expected_b}, received {}/{}.",
                a.len(),
                b.len()
            ));
        }
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let dimensions = [m, k, n]
            .into_iter()
            .map(|dimension| {
                c_int::try_from(dimension)
                    .map_err(|_| "Vulkan matrix dimension exceeds the C API limit.".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut output = vec![0.0; output_len];
        let result = unsafe {
            vk_matmul_checked(
                a.as_ptr(),
                b.as_ptr(),
                output.as_mut_ptr(),
                dimensions[0],
                dimensions[1],
                dimensions[2],
            )
        };
        if result == 0 {
            Ok(output)
        } else {
            Err("Vulkan matrix multiplication failed; see the native diagnostic above.".into())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn matmul_rejects_invalid_shapes_before_calling_vulkan() {
            assert!(matmul(&[], &[], 0, 1, 1).is_err());
            assert!(matmul(&[1.0], &[], 1, 1, 1).is_err());
        }

        #[test]
        fn matmul_returns_native_failure_when_backend_is_uninitialized() {
            assert!(matmul(&[1.0], &[2.0], 1, 1, 1).is_err());
        }

        #[test]
        #[ignore = "requires a Vulkan compute device and compiled matmul.spv"]
        fn matmul_computes_on_vulkan_device() {
            initialize().unwrap();
            let output = matmul(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2).unwrap();
            shutdown().unwrap();
            assert_eq!(output, vec![19.0, 22.0, 43.0, 50.0]);
        }
    }
}

//! TurboQuant-style quantization kernel, compiled to wasm32-unknown-unknown.
//!
//! Exposes the compute-heavy primitive — a randomized fast Walsh–Hadamard
//! transform (a structured random rotation) — plus a fixed scratch buffer that
//! the JS side reads/writes directly in linear memory. No wasm-bindgen, no std:
//! the host marshals f32 arrays into SCRATCH, calls `rotate`, and reads back.
//!
//! The rotation is data-oblivious: identical (seed, n, scale) reproduce the
//! same orthonormal rotation for both query and stored vectors, so inner
//! products are preserved while coordinates become approximately Gaussian —
//! which is what makes per-coordinate scalar quantization near-optimal.

#![no_std]
#![allow(static_mut_refs)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

/// 16,384 f32 slots (64 KiB) — enough for any realistic padded embedding.
const CAP: usize = 1 << 14;
static mut SCRATCH: [f32; CAP] = [0.0; CAP];

/// Pointer to the scratch buffer in linear memory (host writes f32s here).
#[no_mangle]
pub extern "C" fn scratch_ptr() -> *mut f32 {
    unsafe { SCRATCH.as_mut_ptr() }
}

/// Capacity of the scratch buffer, in f32 slots.
#[no_mangle]
pub extern "C" fn scratch_cap() -> usize {
    CAP
}

/// In-place randomized FWHT over the first `n` slots (`n` a power of two ≤ CAP).
///
/// 1. seed-derived random sign flip (xorshift32),
/// 2. fast Walsh–Hadamard butterfly,
/// 3. multiply by `scale` (pass 1/sqrt(n) for an orthonormal transform).
#[no_mangle]
pub extern "C" fn rotate(n: usize, seed: u32, scale: f32) {
    if n == 0 || n > CAP || (n & (n - 1)) != 0 {
        return; // require a non-zero power of two within capacity
    }
    unsafe {
        let buf = &mut SCRATCH;

        // 1. random sign flip
        let mut s = seed | 1;
        for i in 0..n {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if s & 1 == 1 {
                buf[i] = -buf[i];
            }
        }

        // 2. FWHT butterfly
        let mut len = 1usize;
        while len < n {
            let mut i = 0usize;
            while i < n {
                let mut j = i;
                while j < i + len {
                    let a = buf[j];
                    let b = buf[j + len];
                    buf[j] = a + b;
                    buf[j + len] = a - b;
                    j += 1;
                }
                i += len << 1;
            }
            len <<= 1;
        }

        // 3. orthonormal scale
        for i in 0..n {
            buf[i] *= scale;
        }
    }
}

/// Inverse of `rotate` for the same (n, seed, scale): the transpose Rᵀ = D·(sH).
/// Applies the FWHT and scale first, then the same seed-derived sign flip. Since
/// the scaled Hadamard and the sign flip are both involutions, irotate(rotate(v))
/// == v exactly. Used to reconstruct vectors (DeQuant) — `rotate` alone only
/// needs the forward direction for inner-product estimation.
#[no_mangle]
pub extern "C" fn irotate(n: usize, seed: u32, scale: f32) {
    if n == 0 || n > CAP || (n & (n - 1)) != 0 {
        return;
    }
    unsafe {
        let buf = &mut SCRATCH;

        // 1. FWHT butterfly
        let mut len = 1usize;
        while len < n {
            let mut i = 0usize;
            while i < n {
                let mut j = i;
                while j < i + len {
                    let a = buf[j];
                    let b = buf[j + len];
                    buf[j] = a + b;
                    buf[j + len] = a - b;
                    j += 1;
                }
                i += len << 1;
            }
            len <<= 1;
        }

        // 2. orthonormal scale
        for i in 0..n {
            buf[i] *= scale;
        }

        // 3. same seed-derived sign flip (its own inverse)
        let mut s = seed | 1;
        for i in 0..n {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if s & 1 == 1 {
                buf[i] = -buf[i];
            }
        }
    }
}

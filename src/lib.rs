//! Library entry point of the Cheetah command-line application.
//!
//! Cheetah is a local LLM inference tool: it can "prepare" (init) an
//! HF-format model into aligned shards plus a metadata manifest, then run
//! chat inference through the mmap native backend (CPU) or the Vulkan
//! backend (GPU), and also expose an OpenAI-compatible HTTP server.
//!
//! This crate is primarily a binary; the only library module exposed here
//! is `native`, which wraps the FFI surface of the three `cheetah_*.c`
//! translation units:
//!
//! - `cheetah_init.c`   — model preparation (safetensors → aligned
//!   shards / manifest / units)
//! - `cheetah_memory.c` — Ant C manager (units index and tier bookkeeping)
//! - `cheetah_vulkan.c` — Vulkan substrate (device init, VRAM pool, operators)

pub mod native;

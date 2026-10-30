# 🐆 Cheetah

> **Status:** 🔒 Private Development / Phase 0 (Learning & Prototyping)  
> **Target:** Modern Hardware & 100% Pure Rust  
> **Established:** 2026-10-31

## The Philosophy
While **Wolf** 🐺 focuses on lowering the barrier to entry (GGUF, legacy hardware, zero-config), **Cheetah** is built to push the absolute limits of modern hardware.

Cheetah is a highly customizable inference engine and gateway, powered entirely by **HuggingFace's Candle**. 

## Core Objectives
- **Native Format:** Built exclusively for `Safetensors`.
- **Pure Blood:** 100% Rust. No C++ baggage, no legacy `#ifdef` bloat.
- **Hardware Focus:** Fully optimized for Apple Silicon (Metal) and modern GPUs.
- **Distributed Swarm:** Seamless integration with the upcoming **Bee** 🐝 module for distributed local compute.
- **Extreme Customizability:** Expose tensor-level APIs for developers to write custom kernels and forward passes.

## Why a separate project?
Because an engine optimized for a 10-year-old CPU (Wolf) should not dictate the architecture of an engine built for an M3 Max. 

*"To lower the barrier with GGUF, and to push the limits with Safetensors."*

---
*Coming out of the lab when it's ready.*

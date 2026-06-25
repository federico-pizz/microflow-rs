//! [![crates.io](https://img.shields.io/crates/v/microflow)](https://crates.io/crates/microflow)
//! [![docs.rs](https://img.shields.io/docsrs/microflow)](https://docs.rs/microflow)
//! [![github](https://img.shields.io/github/actions/workflow/status/matteocarnelos/microflow-rs/cargo.yml?branch=main)](https://github.com/matteocarnelos/microflow-rs/actions/workflows/cargo.yml)
//!
//! A robust and efficient TinyML inference engine for embedded systems.

#![no_std]

pub use microflow_macros::*;

pub mod activation;
pub mod buffer;
#[cfg(feature = "multicore")]
pub mod multicore;
pub mod ops;
#[cfg(feature = "profiling")]
pub mod profile;
pub mod quantize;
pub mod tensor;

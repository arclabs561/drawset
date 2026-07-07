//! Compatibility crate for the `kuji` to `drawset` rename.
//!
//! New code should depend on `drawset`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use drawset::*;

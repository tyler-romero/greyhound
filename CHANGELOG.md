# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial alpha release of Greyhound as `greyhound-kernels`.
- CuTe DSL kernels exposed through PyTorch custom ops and `greyhound.nn.functional`.
- Core kernel coverage for cross-entropy, chunked linear cross-entropy, causal Conv1D,
  selective log-softmax, and chunked linear loss.
- Experimental bonus Newton-Schulz orthogonalization utility using Quack symmetric GEMM.
- Benchmarking scripts and reference benchmark data for supported kernels and third-party
  provider comparisons.
- Individual documentation pages for each kernel, with kernel design, usage examples, and benchmark plots.

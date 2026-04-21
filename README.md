# MGPP

PyTorch implementation of **MGPP: Multi-Granularity Phase Picking for Seismic Phase Picking**.

## Overview

MGPP is a deep-learning-based seismic phase-picking framework designed to improve robustness and generalization under complex seismic environments.

Unlike conventional methods that mainly rely on single-scale input representations or internal feature pyramids, MGPP explicitly models seismic waveforms from multiple temporal granularities at the input level, enabling the network to jointly capture:

- Local transient waveform characteristics  
- Multi-resolution temporal patterns  
- Long-range contextual dependencies  
- Cross-granularity complementary information  

The framework integrates:

- Channel-Adaptive Patch Embedding (CAPE)
- Multi-Granularity Feature Extraction
- Cross-Granularity State Space Modeling
- Adaptive Fusion Decoder

---

## Highlights

- Strong cross-dataset generalization ability  
- Robust performance under low-SNR conditions  
- Fast inference speed with low GPU memory usage

---

## Architecture

```text
Input Waveform
     ↓
Multi-Granularity Partitioning
     ↓
CAPE Embedding
     ↓
Granularity-specific Feature Extraction
     ↓
Cross-Granularity Mamba Interaction
     ↓
Adaptive Fusion
     ↓
Phase Decoder
     ↓
P / S / Noise Probabilities

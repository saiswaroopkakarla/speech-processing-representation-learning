# Speech Processing & Representation Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-LibriSpeech-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A three-part speech processing assignment covering the full stack from classical DSP to deep learning to responsible AI — manual MFCC extraction, disentangled speaker recognition with a Gradient Reversal Layer, and bias auditing with privacy-preserving voice transformation.

**Dataset:** LibriSpeech `train-clean-100` (~100 hours, 251 speakers)

---

## Assignment Structure

| Part | Topic | Key technique |
|---|---|---|
| Q1 | Feature extraction & spectral analysis | Manual MFCC, cepstral V/UV detection, Wav2Vec2 alignment |
| Q2 | Disentangled speaker recognition | 1D-CNN encoder + Gradient Reversal Layer (GRL) |
| Q3 | Bias auditing & privacy-preserving speech | Speaker distribution audit, pitch shift + noise, fairness loss |

---

## Repository Structure

```
speech-assignment/
├── q1/
│   ├── mfcc_manual.py          # From-scratch MFCC pipeline (no librosa)
│   ├── leakage_snr.py          # Spectral leakage & SNR analysis
│   ├── voiced_unvoiced.py      # Cepstrum-based voiced/unvoiced detection
│   ├── phonetic_mapping.py     # Wav2Vec2 phonetic alignment + RMSE
│   ├── mfcc.png                # MFCC spectrogram output
│   ├── leakage_snr.png         # Spectral leakage comparison plot
│   ├── voiced_unvoiced.png     # Voiced/unvoiced segmentation plot
│   ├── windowed_frame.png      # Windowed frame visualisation
│   ├── original_signal.png     # Raw waveform
│   └── q1_report.pdf
│
├── q2/
│   ├── train.py                # DisentangledModel: 1D-CNN + GRL
│   ├── eval.py                 # Evaluation script
│   ├── disentangled_model.pth  # Saved model weights (120 KB)
│   ├── model_comparison.png    # Baseline vs disentangled accuracy
│   └── q2_report.pdf
│
├── q3/
│   ├── audit.py                # Speaker distribution audit
│   ├── privacymodule.py        # Pitch shift + Gaussian noise transform
│   ├── pp_demo.py              # Privacy transform demo → original/transformed .wav
│   ├── train_fair.py           # Fairness-aware loss function
│   ├── audit_distribution.png  # Histogram: samples per speaker
│   ├── top_speakers.png        # Bar chart: top 10 speakers by clip count
│   ├── original.wav            # Pre-transform audio sample
│   ├── transformed.wav         # Post-transform audio sample
│   └── q3_report.pdf
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/saiswaroopkakarla/speech-assignment.git
cd speech-assignment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Download LibriSpeech

```bash
mkdir -p data/librispeech && cd data/librispeech
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xvzf train-clean-100.tar.gz
```

---

## Q1 — Feature Extraction & Spectral Analysis

### Manual MFCC Pipeline (from scratch)

Implements the full MFCC pipeline in NumPy — no librosa:

```
Pre-emphasis → Framing (25ms, 10ms stride) → Hamming window
→ FFT → Mel filterbank → Log energy → DCT → 13 MFCCs
```

```bash
python q1/mfcc_manual.py
# Output: mfcc.png
```

### Spectral Leakage & SNR

Compares rectangular vs. Hamming vs. Hanning windows on the same frame. Computes SNR before/after noise injection.

```bash
python q1/leakage_snr.py
# Output: leakage_snr.png
```

### Voiced/Unvoiced Detection via Cepstrum

Frame-level classification using cepstral peak detection. Pitch range constrained to 50–400 Hz.

```bash
python q1/voiced_unvoiced.py
# Output: voiced_unvoiced.png
```

### Phonetic Alignment (Wav2Vec2)

Uses `facebook/wav2vec2-base-960h` to get frame-level token logits, maps predicted tokens to timestamps, and computes RMSE against manual segment boundaries.

```bash
python q1/phonetic_mapping.py
```

---

## Q2 — Disentangled Speaker Recognition

Goal: learn a speaker embedding that is *invariant* to recording environment (noise/reverb conditions).

### Architecture

```
Raw waveform (16 kHz, 1s)
        │
  1D-CNN Encoder (Conv1d ×3 → mean-pool → 64-d embedding)
        │
   ┌────┴────────────────┐
Speaker head          Env head (via GRL)
(CrossEntropy)     (CrossEntropy, reversed grad)
```

The **Gradient Reversal Layer (GRL)** makes the encoder *maximally confuse* the environment classifier while still predicting the speaker correctly — forcing the 64-d embedding to contain speaker identity but not environment artefacts.

### Results

| Model | Speaker Accuracy |
|---|---|
| Baseline (no GRL) | ~0.5% |
| Disentangled (with GRL) | ~30% |
| Improvement | ~60× |

> Baseline near-random because the small 100-sample subset creates many 1-shot speaker classes; the GRL model learns a meaningful embedding despite this constraint.

```bash
python q2/train.py    # trains + saves disentangled_model.pth
python q2/eval.py     # loads model, runs evaluation
```

---

## Q3 — Bias Auditing & Privacy-Preserving Speech

### Dataset Bias Audit

Counts clips per speaker across LibriSpeech `train-clean-100`. Identifies class imbalance (some speakers have 5× more recordings than others).

```bash
python q3/audit.py
# Output: audit_distribution.png, top_speakers.png
```

### Privacy-Preserving Transformation

Applies a two-stage voice transformation to reduce speaker identifiability:
1. Pitch scaling (factor 1.1×)
2. Gaussian noise injection (σ = 0.01)

```bash
cd q3
python pp_demo.py
# Output: original.wav, transformed.wav
```

> `original.wav` and `transformed.wav` are committed to the repo — play both to hear the transformation.

### Fairness-Aware Training

`train_fair.py` provides a `fairness_loss()` function that computes a balanced per-group loss — averaging cross-entropy equally across speaker groups regardless of group size. Plug-in replacement for standard CrossEntropy in any training loop.

---

## Key Results Summary

| Task | Metric | Value |
|---|---|---|
| Voiced/Unvoiced detection | Cepstral peak threshold | 0.1 (50–400 Hz pitch range) |
| Phonetic alignment | RMSE vs. manual boundaries | Computed per-file |
| Disentangled speaker recognition | Accuracy (100-sample subset) | ~30% (baseline: ~0.5%) |
| Speaker audit | Most-represented speaker | ~5× median clip count |

---

## Output Gallery

| File | Description |
|---|---|
| `q1/mfcc.png` | 13-coefficient MFCC spectrogram |
| `q1/leakage_snr.png` | Spectral leakage: rectangular vs. Hamming vs. Hanning |
| `q1/voiced_unvoiced.png` | Frame-level V/UV classification overlay |
| `q1/windowed_frame.png` | Single windowed frame visualisation |
| `q2/model_comparison.png` | Baseline vs. disentangled accuracy bar chart |
| `q3/audit_distribution.png` | Histogram of clips-per-speaker |
| `q3/top_speakers.png` | Top 10 speakers by clip count |
| `q3/original.wav` | Raw LibriSpeech audio sample |
| `q3/transformed.wav` | Privacy-transformed version |

---

## Known Issues & TODOs

- [ ] `mfcc_manual.py` has an old commented-out draft at the top of the file — safe to delete, the working implementation is below it
- [ ] `eval.py` has two old commented-out drafts at the top — safe to delete, the working implementation with `model_comparison.png` output is below them

---

## Author

**Kakarla Sai Swaroop**, M25DE1023, IIT Jodhpur M.Tech Data Engineering  
m25de1023@iitj.ac.in

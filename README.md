# 🎧 ONNX Audio Denoiser (Demucs-based)

A **Python-based audio denoising pipeline** built on top of an ONNX-exported
Demucs-style model. This project was migrated from a C# backend implementation
to Python to enable **model inspection, experimentation, and future ML
improvements**.

---

## ✨ Features

- ✅ ONNX Runtime inference (CPU)
- ✅ 16 kHz mono audio denoising
- ✅ Demucs-style padding (`valid_length`)
- ✅ MP3 / WAV input support
- ✅ Environment-based configuration (`.env`)
- ✅ Reproducible builds using `uv`

---

## 📁 Project Structure

```text
ml_audio_denoiser/
├── models/
│   └── demucs_16k_dynamic.onnx
├── scripts/
│   ├── audio_denoiser.py
│   └── test_denoiser.py
├── audio/                # (ignored in git)
├── pyproject.toml
├── uv.lock
├── .env                  # (ignored in git)
└── README.md

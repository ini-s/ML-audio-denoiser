from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort


class AudioDenoiser:
    """
    - Expects 16 kHz mono float32 waveform in range [-1, 1]
    - Pads to a 'valid length' (Demucs-like), runs ONNX, trims back to original
    """

    def __init__(self, model_path: str, providers: Optional[list[str]] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        if providers is None:
            # CPU by default (works everywhere)
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Be robust to model I/O names; prefer standard names if present
        self.input_name, self.output_name = self._pick_io_names()

    # ---- Mirror of Demucs.valid_length ----
    @staticmethod
    def valid_length(
        length: int, depth: int = 5, kernel: int = 8, stride: int = 4, resample: int = 4
    ) -> int:
        l = int(np.ceil(length * float(resample)))
        #encoder - downsampling path
        for _ in range(depth):
            l = int(np.ceil((l - kernel) / float(stride))) + 1
            l = max(l, 1)
        #decoder - upsampling path
        for _ in range(depth):
            l = (l - 1) * stride + kernel
        return int(np.ceil(l / float(resample)))


    def denoise(self, waveform: np.ndarray) -> np.ndarray:
        if waveform is None or waveform.size == 0:
            return np.array([], dtype=np.float32)

        if waveform.ndim != 1:
            raise ValueError(f"Expected mono 1-D waveform, got shape {waveform.shape}")

        x = waveform.astype(np.float32, copy=False)
        t_orig = x.shape[0]
        
        #length adjusted to the nearest valid length for the model
        t_valid = self.valid_length(t_orig)

        if t_valid != t_orig:
            padded = np.zeros((t_valid,), dtype=np.float32)
            padded[:t_orig] = x
        else:
            padded = x

        # ONNX expects [B=1, C=1, T] - [batch, channels, time]
        inp = padded[None, None, :]  # shape (1, 1, T)

        outputs = self.session.run([self.output_name], {self.input_name: inp})
        y = outputs[0]  # expect shape (1, 1, T) or (1, 1, >=T_valid)

        if y.ndim != 3 or y.shape[0] != 1 or y.shape[1] != 1:
            raise RuntimeError(f"Unexpected output shape {y.shape}; expected (1,1,T)")

        denoised = y[0, 0, :t_orig].astype(np.float32, copy=False)
        # Return a contiguous copy (safe for consumers)
        return np.copy(denoised)

    #Tries to explicitly close underlying session handles (not strictly necessary; Python GC will clean up).
    def close(self) -> None:
        try:
            self.session._sess.disconnect()  
        except Exception:
            pass

    def __del__(self):
        try:
            self.session = None
        except Exception:
            pass

    # ---- Helpers ----
    def _pick_io_names(self) -> Tuple[str, str]:
        ins = self.session.get_inputs()
        outs = self.session.get_outputs()
        if not ins or not outs:
            raise RuntimeError("Model must have at least one input and one output.")

        # Try common names first, else fall back to first tensors
        def pick(candidates, available):
            names = [a.name for a in available]
            for c in candidates:
                for n in names:
                    if n.lower() == c:
                        return n
            return names[0]

        input_name = pick(["input", "inputs", "model_input", "input_0"], ins)
        output_name = pick(["output", "outputs", "model_output", "output_0"], outs)

        return input_name, output_name

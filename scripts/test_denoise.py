import os
import time
import numpy as np
import soundfile as sf
import librosa
from dotenv import load_dotenv

from audio_denoiser import AudioDenoiser

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
INPUT_AUDIO = os.getenv("INPUT_AUDIO")
OUTPUT_AUDIO = os.getenv("OUTPUT_AUDIO")


def load_and_prepare(path):
    # Load MP3 or any audio format , sr - sample rate
    audio, sr = librosa.load(path, sr=None, mono=False)

    # Ensure mono
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    # Ensure 16kHz sample rate
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Must be float32 for ONNX
    audio = audio.astype(np.float32)

    return audio, sr


def main():
    print("Loading audio:", INPUT_AUDIO)
    audio, sr = load_and_prepare(INPUT_AUDIO)

    print("Loading ONNX model…")
    denoiser = AudioDenoiser(MODEL_PATH)

    print("Running denoising…")
    start_time = time.time()              
    cleaned = denoiser.denoise(audio)
    end_time = time.time()               

    elapsed = end_time - start_time

    print(f"\n⏱️  Denoising Time: {elapsed:.3f} seconds ({elapsed*1000:.2f} ms)")

    print("Saving output:", OUTPUT_AUDIO)
    sf.write(OUTPUT_AUDIO, cleaned, 16000)

    print("\n🎉 Done! Cleaned file saved at:")
    print(OUTPUT_AUDIO)


if __name__ == "__main__":
    main()
import glob
import os
import torch
import torchaudio
import numpy as np

from pathlib import Path


def run(original_path: str):
    wavs = glob.glob(os.path.join(original_path, '*.wav'))
    for wav_path in wavs:
        name = Path(wav_path).stem
        wav, sr = torchaudio.load(wav_path)
        wav = wav.cpu().numpy().squeeze()
        original_wav_size = len(name)
        txt = Path(original_path) / f"{name}_is_lost.txt"
        mask = np.array([1 - np.array(list(map(int, open(txt, 'r').read().strip('\n').split('\n'))))])
        mask = torch.repeat_interleave(torch.tensor(mask, dtype=torch.float32), 320)
        cut_wav_size = (original_wav_size // 320) * 320
        wav = wav[:cut_wav_size]
        if mask.shape[0] < cut_wav_size:
            print(f"Expected mask to be bigger or same than cut_wav_size, got {mask.shape[0]}, "
                  f"{cut_wav_size}, sample {wav_path}")


if __name__ == "__main__":
    original_path = "blind/lossy_signals/"
    run(original_path)

import glob
import os
import numpy as np
from pathlib import Path
import torchaudio
from PLCMOS.plc_mos import PLCMOSEstimator
from utils.utils import LSD
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI


def run(source_path, ref_path):
    plcmos = PLCMOSEstimator()
    intrusives = []
    non_intrusives = []
    lsds = []
    stoi = []

    stoi_model = STOI(16000)
    wavs = glob.glob(os.path.join(source_path, '*.wav'))
    for wav_path in wavs:
        name = Path(wav_path).name
        try:
            wav, sr = torchaudio.load(wav_path)
        except RuntimeError:
            print(f"Error opening {wav_path}")
            continue
        wav = wav.cpu().squeeze()
        ref_wav_path = Path(ref_path) / name
        try:
            ref_wav, ref_sr = torchaudio.load(ref_wav_path)
        except RuntimeError:
            print(f"Error opening {ref_wav_path}")
            continue
        ref_wav = ref_wav.cpu().squeeze()
        min_shape = min(len(ref_wav), len(wav))
        wav = wav[:min_shape]
        ref_wav = ref_wav[:min_shape]
        assert ref_sr and sr == 16000, f"Exprected sr to be 16000, got: {sr}, {ref_sr}"
        stoi.append(stoi_model(wav, ref_wav))
        ret = plcmos.run(wav.numpy(), ref_wav.numpy())
        intrusives.append(ret[0])
        non_intrusives.append(ret[1])
        lsds.append(LSD(ref_wav.numpy(), wav.numpy())[0])
        print(f"{stoi[-1]}, {ret[0]}, {ret[1]}")

    log_dict = {
        "Intrusive": float(np.mean(intrusives)),
        "Non-intrusive": float(np.mean(non_intrusives)),
        'LSD': float(np.mean(lsds)),
        'STOI': float(np.mean(stoi)),
    }
    print(log_dict)


if __name__ == "__main__":
    source_path = "lpcnet_out"
    ref_path = "X_CleanReference"
    run(source_path, ref_path)

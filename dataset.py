import glob
import os
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from numpy.random import default_rng
from pydtmc import MarkovChain
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import CONFIG

np.random.seed(0)
rng = default_rng()


def load_audio(
        path,
        sample_rate: int = 16000,
        chunk_len=None,
):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        audio_len = f.frames

        if chunk_len is not None and chunk_len < audio_len:
            start_index = torch.randint(0, audio_len - chunk_len, (1,))[0]
            frames = f._prepare_read(start_index, start_index + chunk_len, -1)
            audio = f.read(frames, always_2d=True, dtype="float32")

        else:
            audio = f.read(always_2d=True, dtype="float32")

    if sr != sample_rate:
        audio = librosa.resample(np.squeeze(audio), orig_sr=sr, target_sr=sample_rate)[:, np.newaxis]

    return audio.T


def pad(sig, length):
    if sig.shape[1] < length:
        pad_len = length - sig.shape[1]
        sig = torch.hstack((sig, torch.zeros((sig.shape[0], pad_len))))

    else:
        start = random.randint(0, sig.shape[1] - length)
        sig = sig[:, start:start + length]
    return sig


def pad_mask(mask, target_size):
    pass


class MaskGenerator:
    def __init__(self, is_train=True, probs=((0.9, 0.1), (0.5, 0.1), (0.5, 0.5))):
        '''
            is_train: if True, mask generator for training otherwise for evaluation
            probs: a list of transition probability (p_N, p_L) for Markov Chain. Only allow 1 tuple if 'is_train=False'
        '''
        self.is_train = is_train
        self.probs = probs
        self.mcs = []
        if self.is_train:
            for prob in probs:
                self.mcs.append(MarkovChain([[prob[0], 1 - prob[0]], [1 - prob[1], prob[1]]], ['1', '0']))
        else:
            assert len(probs) == 1
            prob = self.probs[0]
            self.mcs.append(MarkovChain([[prob[0], 1 - prob[0]], [1 - prob[1], prob[1]]], ['1', '0']))

    def gen_mask(self, length, seed=0):
        if self.is_train:
            mc = random.choice(self.mcs)
        else:
            mc = self.mcs[0]
        mask = mc.walk(length - 1, seed=seed)
        mask = np.array(list(map(int, mask)))
        return mask


class TestLoader(Dataset):
    def __init__(self):
        dataset_name = CONFIG.DATA.dataset
        self.mask = CONFIG.DATA.EVAL.masking

        self.target_root = CONFIG.DATA.data_dir[dataset_name]['root']
        txt_list = CONFIG.DATA.data_dir[dataset_name]['test']
        self.data_list = self.load_txt(txt_list)
        if self.mask == 'real':
            trace_txt = glob.glob(os.path.join(CONFIG.DATA.EVAL.trace_path, '*.txt'))
            trace_txt.sort()
            self.trace_list = [1 - np.array(list(map(int, open(txt, 'r').read().strip('\n').split('\n')))) for txt in
                               trace_txt]
        else:
            self.mask_generator = MaskGenerator(is_train=False, probs=CONFIG.DATA.EVAL.transition_probs)

        self.sr = CONFIG.DATA.sr
        self.stride = CONFIG.DATA.stride
        self.window_size = CONFIG.DATA.window_size
        self.audio_chunk_len = CONFIG.DATA.audio_chunk_len
        self.p_size = CONFIG.DATA.EVAL.packet_size  # 20ms
        self.hann = torch.sqrt(torch.hann_window(self.window_size))

    def __len__(self):
        return len(self.data_list)

    def load_txt(self, txt_list):
        target = []
        with open(txt_list) as f:
            for line in f:
                target.append(os.path.join(self.target_root, line.strip('\n')))
        target = list(set(target))
        target.sort()
        return target

    def __getitem__(self, index):
        target = load_audio(self.data_list[index], sample_rate=self.sr)
        target = target[:, :(target.shape[1] // self.p_size) * self.p_size]

        sig = np.reshape(target, (-1, self.p_size)).copy()
        if self.mask == 'real':
            mask = self.trace_list[index % len(self.trace_list)]
            mask = np.repeat(mask, np.ceil(len(sig) / len(mask)), 0)[:len(sig)][:, np.newaxis]
        else:
            mask = self.mask_generator.gen_mask(len(sig), seed=index)[:, np.newaxis]
        sig *= mask
        sig = torch.tensor(sig).reshape(-1)

        target = torch.tensor(target).squeeze(0)

        sig_wav = sig.clone()
        target_wav = target.clone()

        target = torch.view_as_real(torch.stft(target, self.window_size, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1)
        sig = torch.view_as_real(torch.stft(sig, self.window_size, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1)
        return sig.float(), target.float(), sig_wav, target_wav, mask


class SaintyCheckLoader(Dataset):
    def __init__(self, size=None):
        self.size = size
        self.mask = CONFIG.NBTEST.masking
        self.data_list = glob.glob(os.path.join(CONFIG.NBTEST.real_dir, '*.wav'))
        if self.mask == 'real':
            trace_txt = glob.glob(os.path.join(CONFIG.NBTEST.loss_path, '*.txt'))
            self.trace_list = [1 - np.array(list(map(int, open(txt, 'r').read().strip('\n').split('\n')))) for txt in
                               trace_txt]
            assert len(self.trace_list) == len(self.data_list)
        else:
            self.mask_generator = MaskGenerator(is_train=False, probs=CONFIG.NBTEST.transition_probs)

        self.sr = CONFIG.DATA.sr
        self.stride = CONFIG.DATA.stride
        self.window_size = CONFIG.DATA.window_size
        self.p_size = CONFIG.NBTEST.packet_size  # 20ms
        self.hann = torch.sqrt(torch.hann_window(self.window_size))
        self.mask_generator = MaskGenerator(is_train=True, probs=CONFIG.DATA.TRAIN.transition_probs)

    def __len__(self):
        return self.size if self.size is not None else len(self.data_list)

    def _create_mask_for_causal_inference(self, mask, sig):
        """
        mask: np.ndarray shape (audio_len // p_size)
        sig: torch.FloatTensor, stft res shape C, F, T
        return: mask of shape C, F, T, where original mask is twice repeated and padded if needed along T dim
        """
        mask = torch.tensor(mask, dtype=torch.long).squeeze(-1)
        C, F, T = sig.shape
        mask_target_size = mask.shape[0] * 2
        mask = mask.repeat(2, 1).T.reshape(-1)
        if T - mask.shape[0] == 1:
            mask = torch.cat([mask, torch.tensor([0])])
        reshaped = torch.repeat_interleave(mask, F, dim=0).reshape(T, F)
        reshaped = torch.stack([reshaped for _ in range(C)]).permute(0, 2, 1)
        return reshaped.detach()

    def __getitem__(self, index):
        target = load_audio(self.data_list[index], sample_rate=self.sr)
        target = target[:, :(target.shape[1] // self.p_size) * self.p_size]
        sig = np.reshape(target, (-1, self.p_size)).copy()
        target = np.reshape(target, (-1, self.p_size)).copy()
        if self.mask == 'real':
            mask = self.trace_list[index][:, np.newaxis]
            assert sig.shape[0] == mask.shape[0]
        else:
            mask = self.mask_generator.gen_mask(len(sig), seed=index)[:, np.newaxis]
        sig *= mask
        sig = torch.tensor(sig).reshape(-1)
        target = torch.tensor(target).reshape(-1)
        target = torch.view_as_real(torch.stft(target, self.window_size, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1)
        sig = torch.view_as_real(torch.stft(sig, self.window_size, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1)
        reshaped_mask = self._create_mask_for_causal_inference(mask, sig)
        return sig.float(), target.float(), reshaped_mask


class NonBlindTestLoader(Dataset):
    def __init__(self, size=None):
        self.size = size
        self.mask = CONFIG.NBTEST.masking
        self.data_list = glob.glob(os.path.join(CONFIG.NBTEST.real_dir, '*.wav'))
        if self.mask == 'real':
            trace_txt = glob.glob(os.path.join(CONFIG.NBTEST.loss_path, '*.txt'))
            self.trace_list = [1 - np.array(list(map(int, open(txt, 'r').read().strip('\n').split('\n')))) for txt in
                               trace_txt]
            assert len(self.trace_list) == len(self.data_list)
        else:
            self.mask_generator = MaskGenerator(is_train=False, probs=CONFIG.NBTEST.transition_probs)

        self.sr = CONFIG.DATA.sr
        self.stride = CONFIG.DATA.stride
        self.window_size = CONFIG.DATA.window_size
        self.p_size = CONFIG.NBTEST.packet_size  # 20ms
        self.hann = torch.sqrt(torch.hann_window(self.window_size))
        self.mask_generator = MaskGenerator(is_train=True, probs=CONFIG.DATA.TRAIN.transition_probs)

    def __len__(self):
        return self.size if self.size is not None else len(self.data_list)

    def _create_mask_for_causal_inference(self, mask, sig):
        """
        mask: np.ndarray shape (audio_len // p_size)
        sig: torch.FloatTensor, stft res shape C, F, T
        return: mask of shape C, F, T, where original mask is twice repeated and padded if needed along T dim
        """
        mask = torch.tensor(mask, dtype=torch.long).squeeze(-1)
        C, F, T = sig.shape
        mask_target_size = mask.shape[0] * 2
        mask = mask.repeat(2, 1).T.reshape(-1)
        if T - mask.shape[0] == 1:
            mask = torch.cat([mask, torch.tensor([0])])
        reshaped = torch.repeat_interleave(mask, F, dim=0).reshape(T, F)
        reshaped = torch.stack([reshaped for _ in range(C)]).permute(0, 2, 1)
        return reshaped.detach()

    def __getitem__(self, index):
        target = load_audio(self.data_list[index], sample_rate=self.sr)
        target = target[:, :(target.shape[1] // self.p_size) * self.p_size]
        sig = np.reshape(target, (-1, self.p_size)).copy()
        target = np.reshape(target, (-1, self.p_size)).copy()
        if self.mask == 'real':
            mask = self.trace_list[index][:, np.newaxis]
            assert sig.shape[0] == mask.shape[0]
        else:
            mask = self.mask_generator.gen_mask(len(sig), seed=index)[:, np.newaxis]
        sig *= mask
        sig = torch.tensor(sig).reshape(-1)
        target = torch.tensor(target).reshape(-1)
        sig_wav = sig.clone()
        target_wav = target.clone()
        target = torch.view_as_real(torch.stft(target, self.window_size, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1)
        sig = torch.view_as_real(torch.stft(sig, self.window_size, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1)
        reshaped_mask = self._create_mask_for_causal_inference(mask, sig)
        return sig.float(), target.float(), sig_wav, target_wav, reshaped_mask


class BlindTestLoader(Dataset):
    def __init__(self, test_dir, compute_metrics: bool = False):
        self.data_list = glob.glob(os.path.join(test_dir, '*.wav'))[:3]
        self.real_data_dir = CONFIG.TEST.real_dir
        self.sr = CONFIG.DATA.sr
        self.stride = CONFIG.DATA.stride
        self.chunk_len = CONFIG.DATA.window_size
        self.hann = torch.sqrt(torch.hann_window(self.chunk_len))
        self.compute_metrics = compute_metrics

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sig = load_audio(self.data_list[index], sample_rate=self.sr)
        sig = torch.from_numpy(sig).squeeze(0)
        sig_wav = sig.clone()
        sig = torch.view_as_real(
            torch.stft(sig, self.chunk_len, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1)
        if self.compute_metrics:
            filename = Path(self.data_list[index]).name
            target = load_audio(os.path.join(self.real_data_dir, filename))
            target = torch.from_numpy(target).squeeze(0)
            target_wav = target.clone()
            target = torch.view_as_real(torch.stft(target, self.chunk_len, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1)
            return sig.float(), target.float(), sig_wav, target_wav
        else:
            return sig.float(), sig_wav


class TrainDataset(Dataset):

    def __init__(self, mode='train'):
        dataset_name = CONFIG.DATA.dataset
        self.target_root = CONFIG.DATA.data_dir[dataset_name]['root']

        txt_list = CONFIG.DATA.data_dir[dataset_name]['train']
        self.data_list = self.load_txt(txt_list)

        if mode == 'train':
            self.data_list, _ = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

        elif mode == 'val':
            _, self.data_list = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

        self.p_sizes = CONFIG.DATA.TRAIN.packet_sizes
        self.mode = mode
        self.sr = CONFIG.DATA.sr
        self.window = CONFIG.DATA.audio_chunk_len
        self.stride = CONFIG.DATA.stride
        self.chunk_len = CONFIG.DATA.window_size
        self.hann = torch.sqrt(torch.hann_window(self.chunk_len))
        self.mask_generator = MaskGenerator(is_train=True, probs=CONFIG.DATA.TRAIN.transition_probs)

    def __len__(self):
        return len(self.data_list)

    def load_txt(self, txt_list):
        target = []
        with open(txt_list) as f:
            for line in f:
                target.append(os.path.join(self.target_root, line.strip('\n')))
        target = list(set(target))
        target.sort()
        return target

    def fetch_audio(self, index):
        sig = load_audio(self.data_list[index], sample_rate=self.sr, chunk_len=self.window)
        while sig.shape[1] < self.window:
            idx = torch.randint(0, len(self.data_list), (1,))[0]
            pad_len = self.window - sig.shape[1]
            if pad_len < 0.02 * self.sr:
                padding = np.zeros((1, pad_len), dtype=np.float)
            else:
                padding = load_audio(self.data_list[idx], sample_rate=self.sr, chunk_len=pad_len)
            sig = np.hstack((sig, padding))
        return sig

    def _create_mask_for_causal_inference(self, mask, sig):
        """
        mask: np.ndarray shape (audio_len // p_size)
        sig: torch.FloatTensor, stft res shape C, F, T
        return: mask of shape C, F, T, where original mask is twice repeated and padded if needed along T dim
        """
        mask = torch.tensor(mask, dtype=torch.long).squeeze(-1)
        C, F, T = sig.shape
        mask_target_size = mask.shape[0] * 2
        mask = mask.repeat(2, 1).T.reshape(-1)
        if T - mask.shape[0] == 1:
            mask = torch.cat([mask, torch.tensor([0])])
        reshaped = torch.repeat_interleave(mask, F, dim=0).reshape(T, F)
        reshaped = torch.stack([reshaped for _ in range(C)]).permute(0, 2, 1)
        return reshaped.detach()

    def __getitem__(self, index):
        sig = self.fetch_audio(index)

        sig = sig.reshape(-1).astype(np.float32)

        target = torch.tensor(sig.copy())
        p_size = random.choice(self.p_sizes)

        sig = np.reshape(sig, (-1, p_size))
        mask = self.mask_generator.gen_mask(len(sig), seed=index)[:, np.newaxis]
        sig *= mask
        sig = torch.tensor(sig.copy()).reshape(-1)

        target = torch.view_as_real(
            torch.stft(target, self.chunk_len, self.stride, window=self.hann, return_complex=True)).permute(2, 0, 1).float()
        sig = torch.view_as_real(torch.stft(sig, self.chunk_len, self.stride, window=self.hann, return_complex=True))
        sig = sig.permute(2, 0, 1).float()
        reshaped_mask = self._create_mask_for_causal_inference(mask, sig)
        return sig, target, reshaped_mask

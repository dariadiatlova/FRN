import os

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import wandb
from torch import nn
# from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI

from PLCMOS.plc_mos import PLCMOSEstimator
from config import CONFIG
from loss import Loss
from models.blocks import Encoder, Predictor
from utils.utils import visualize, LSD
from utils.tblogger import WandbSpectrogramLogging

plcmos = PLCMOSEstimator()


class PLCModel(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None, window_size=960, enc_layers=4, enc_in_dim=384, enc_dim=768,
                 pred_dim=512, pred_layers=1, pred_ckpt_path='lightning_logs/predictor/checkpoints/predictor.ckpt',
                 lr=None):
        super(PLCModel, self).__init__()
        self.window_size = window_size
        self.hop_size = window_size // 2
        self.learning_rate = lr if lr is not None else CONFIG.TRAIN.lr
        self.hparams.batch_size = CONFIG.TRAIN.batch_size

        self.enc_layers = enc_layers
        self.enc_in_dim = enc_in_dim
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim
        self.pred_layers = pred_layers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.stoi = STOI(48000)
        # self.pesq = PESQ(16000, 'wb')
        self.log_spec = WandbSpectrogramLogging()

        if pred_ckpt_path is not None:
            self.predictor = Predictor.load_from_checkpoint(pred_ckpt_path)
        else:
            self.predictor = Predictor(window_size=self.window_size, lstm_dim=self.pred_dim,
                                       lstm_layers=self.pred_layers)
        self.joiner = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(9, 1), stride=1, padding=(4, 0), padding_mode='reflect',
                      groups=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 2, kernel_size=1, stride=1, padding=0, groups=2),
        )

        self.encoder = Encoder(in_dim=self.window_size, dim=self.enc_in_dim, depth=self.enc_layers,
                               mlp_dim=self.enc_dim)

        self.loss = Loss()
        self.window = torch.sqrt(torch.hann_window(self.window_size))
        self.save_hyperparameters('window_size', 'enc_layers', 'enc_in_dim', 'enc_dim', 'pred_dim', 'pred_layers')

    def forward(self, x, mask):
        #     """
        #     Input: real-imaginary; shape (B, 2, F, T); F = hop_size
        #            mask; shape (B, seq_len, p_dim)
        #     Output: real-imaginary
        #     """
        assert x is None, f"{x.shape}"
        B, C, F, T = x.shape
        mask = mask.permute(3, 0, 1, 2).unsqueeze(-1)
        x = x.permute(3, 0, 1, 2).unsqueeze(-1)
        prev_mag = torch.zeros((B, 1, F, 1), device=x.device)
        predictor_state = torch.zeros((2, self.predictor.lstm_layers, B, self.predictor.lstm_dim),
                                      device=x.device)
        mlp_state = torch.zeros((self.encoder.depth, 2, 1, B, self.encoder.dim), device=x.device)
        result = []
        for i in range(T):
            step = x[i].to(self.device)
            feat, mlp_state = self.encoder(step, mlp_state)
            assert prev_mag is None, f"{prev_mag.shape}, {predictor_state.shape}"
            prev_mag, predictor_state = self.predictor(prev_mag, predictor_state)
            assert prev_mag is None, f"{feat.shape}, {prev_mag.shape}, {x.shape}, {predictor_state.shape}" \
                                     f"{self.predictor}"
            feat = torch.cat((feat, prev_mag), 1)
            feat = self.joiner(feat)
            feat = feat + step
            # if packet is not lost use gt
            feat = torch.where(mask[i] == True, x[i], feat)
            result.append(feat)
            prev_mag = torch.linalg.norm(feat, dim=1, ord=1, keepdims=True)  # compute magnitude
        output = torch.cat(result, -1)
        return output

    def forward_onnx(self, x, prev_mag, predictor_state=None, mlp_state=None):
        prev_mag, predictor_state = self.predictor(prev_mag, predictor_state)
        feat, mlp_state = self.encoder(x, mlp_state)

        feat = torch.cat((feat, prev_mag), 1)
        feat = self.joiner(feat)
        prev_mag = torch.linalg.norm(feat, dim=1, ord=1, keepdims=True)
        feat = feat + x
        return feat, prev_mag, predictor_state, mlp_state

    def training_step(self, batch, batch_idx):
        inp, tar, mask = batch
        f_0 = inp[:, :, 0:1, :]
        x = inp[:, :, 1:, :]
        mask = mask[:, :, 1:, :]
        x = self(x, mask)
        x = torch.cat([f_0, x], dim=2)
        loss = self.loss(x, tar)
        log_dict = {"train_stft_loss": loss.item(), "optimizer_rate/optimizer": self.optimizer.param_groups[0]['lr']}
        self.log_dict(log_dict, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        assert val_batch is None, f"{val_batch[0].shape}"
        inp, tar, mask = val_batch
        f_0 = inp[:, :, 0:1, :]
        x_in = inp[:, :, 1:, :]
        mask = mask[:, :, 1:, :]

        pred = self(x_in, mask)
        pred = torch.cat([f_0, pred], dim=2)

        loss = self.loss(pred, tar)
        self.window = self.window.to(pred.device)
        pred = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())
        pred = torch.istft(pred, self.window_size, self.hop_size, window=self.window)
        y = torch.view_as_complex(tar.permute(0, 2, 3, 1).contiguous())
        y = torch.istft(y, self.window_size, self.hop_size, window=self.window)
        x = torch.view_as_complex(inp.permute(0, 2, 3, 1).contiguous())
        x = torch.istft(x, self.window_size, self.hop_size, window=self.window)

        if batch_idx == 0:
            for i in range(CONFIG.WANDB.log_n_audios):
                self.logger.experiment.log(
                    {f"{i}/original_audio": wandb.Audio(y[i].detach().cpu(),
                                                        caption=f"original_{i}",
                                                        sample_rate=CONFIG.DATA.sr)}
                )

                self.logger.experiment.log(
                    {f"{i}/loosy_audio": wandb.Audio(x[i].detach().cpu(),
                                                     caption=f"loosy_{i}",
                                                     sample_rate=CONFIG.DATA.sr)}
                )

                self.logger.experiment.log(
                    {f"{i}/reconstructed_audio": wandb.Audio(pred[i].detach().cpu(),
                                                             caption=f"reconstructed_{i}",
                                                             sample_rate=CONFIG.DATA.sr)}
                )

                spectrograms = self.log_spec.plot_spectrogram_to_numpy(y[i].detach().cpu(),
                                                                       x[i].detach().cpu(),
                                                                       pred[i].detach().cpu(),
                                                                       self.global_step)
                self.logger.experiment.log({f"Spectrograms_{i}": wandb.Image(spectrograms, caption=f"Spectrogram {i}")})
        stoi = self.stoi(pred, y)
        if CONFIG.DATA.sr != 16000:
            pred, y = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
            pred = librosa.resample(pred, orig_sr=48000, target_sr=16000)
            y = librosa.resample(y, orig_sr=48000, target_sr=16000, res_type='kaiser_fast')
        # LOG METRICS
        # pesq = self.pesq(torch.tensor(pred), torch.tensor(y))
        intrusives = []
        non_intrusives = []
        lsds = []
        for i in range(pred.shape[0]):
            ret = plcmos.run(pred[i, :], y[i, :])
            intrusives.append(ret[0])
            non_intrusives.append(ret[1])
            lsds.append(LSD(y[i, :], pred[i, :])[0])
        log_dict = {
            "Intrusive": float(np.mean(intrusives)),
            "Non-intrusive": float(np.mean(non_intrusives)),
            'LSD': float(np.mean(lsds)),
            'STOI': stoi,
            # 'PESQ': pesq,
            "val_stft_loss": loss.item()
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        inp, tar, inp_wav, tar_wav = test_batch
        inp_wav = inp_wav.squeeze()
        tar_wav = tar_wav.squeeze()
        f_0 = inp[:, :, 0:1, :]
        x = inp[:, :, 1:, :]
        pred = self(x)
        pred = torch.cat([f_0, pred], dim=2)
        pred = torch.istft(pred.squeeze(0).permute(1, 2, 0), self.window_size, self.hop_size,
                           window=self.window.to(pred.device))
        stoi = self.stoi(pred, tar_wav)

        tar_wav = tar_wav.cpu().numpy()
        inp_wav = inp_wav.cpu().numpy()
        pred = pred.detach().cpu().numpy()
        lsd, _ = LSD(tar_wav, pred)

        if batch_idx in [5, 7, 9]:
            sample_path = os.path.join(CONFIG.LOG.sample_path)
            path = os.path.join(sample_path, 'sample_' + str(batch_idx))
            visualize(tar_wav, inp_wav, pred, path)
            sf.write(os.path.join(path, 'enhanced_output.wav'), pred, samplerate=CONFIG.DATA.sr, subtype='PCM_16')
            sf.write(os.path.join(path, 'lossy_input.wav'), inp_wav, samplerate=CONFIG.DATA.sr, subtype='PCM_16')
            sf.write(os.path.join(path, 'target.wav'), tar_wav, samplerate=CONFIG.DATA.sr, subtype='PCM_16')
        if CONFIG.DATA.sr != 16000:
            pred = librosa.resample(pred, orig_sr=48000, target_sr=16000)
            tar_wav = librosa.resample(tar_wav, orig_sr=48000, target_sr=16000, res_type='kaiser_fast')
        ret = plcmos.run(pred, tar_wav)
        # pesq = self.pesq(torch.tensor(pred), torch.tensor(tar_wav))
        metrics = {
            "Intrusive": ret[0],
            "Non-intrusive": ret[1],
            'LSD': lsd,
            'STOI': stoi,
            # 'PESQ': pesq,
        }
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inp, tar, inp_wav, tar_wav, mask = batch
        f_0 = inp[:, :, 0:1, :]
        x_in = inp[:, :, 1:, :]
        mask = mask[:, :, 1:, :]
        pred = self(x_in, mask)
        pred = torch.cat([f_0, pred], dim=2)
        pred = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())
        pred = torch.istft(pred, self.window_size, self.hop_size, window=self.window.to(pred.device))
        return pred, inp_wav

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=CONFIG.TRAIN.patience,
        #                                                           factor=CONFIG.TRAIN.factor, verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CONFIG.TRAIN.epochs,
                                                                  eta_min=1e-8, verbose=True)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=9e-1, last_epoch=CONFIG.TRAIN.epochs)

        # scheduler = {
        #     'scheduler': lr_scheduler,
        #     # 'reduce_on_plateau': True,
        #     'monitor': CONFIG.WANDB.monitor
        # }
        return [self.optimizer], [lr_scheduler]


class OnnxWrapper(pl.LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        batch_size = 1
        pred_states = torch.zeros((2, 1, batch_size, model.predictor.lstm_dim))
        mlp_states = torch.zeros((model.encoder.depth, 2, 1, batch_size, model.encoder.dim))
        mag = torch.zeros((batch_size, 1, model.hop_size, 1))
        x = torch.randn(batch_size, model.hop_size + 1, 2)
        self.sample = (x, mag, pred_states, mlp_states)
        self.input_names = ['input', 'mag_in_cached_', 'pred_state_in_cached_', 'mlp_state_in_cached_']
        self.output_names = ['output', 'mag_out_cached_', 'pred_state_out_cached_', 'mlp_state_out_cached_']

    def forward(self, x, prev_mag, predictor_state=None, mlp_state=None):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        f_0 = x[:, :, 0:1, :]
        x = x[:, :, 1:, :]

        output, prev_mag, predictor_state, mlp_state = self.model.forward_onnx(x, prev_mag, predictor_state, mlp_state)
        output = torch.cat([f_0, output], dim=2)
        output = output.squeeze(-1).permute(0, 2, 1)
        return output, prev_mag, predictor_state, mlp_state

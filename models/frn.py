import itertools
from pathlib import Path

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import wandb
from torch import nn
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI

from PLCMOS.plc_mos import PLCMOSEstimator
from config import CONFIG
from loss import Loss, FMLoss, LeastSquaresDiscLoss, LeastSquaresGenLoss
from models.blocks import Encoder, Predictor
from models.melgan import MultiScaleDiscriminator
from utils.utils import visualize, LSD
from utils.logger import WandbSpectrogramLogging

plcmos = PLCMOSEstimator()


class PLCModel(pl.LightningModule):
    def __init__(self, config):
        super(PLCModel, self).__init__()
        self.config = config
        pred_ckpt_path = CONFIG.TRAIN.pred_ckpt_path
        self.window_size = CONFIG.DATA.window_size
        self.hop_size = CONFIG.DATA.stride
        self.hparams.batch_size = CONFIG.TRAIN.batch_size

        self.learning_rate = float(config["lr"])
        self.enc_layers = int(config["enc_layers"])
        self.enc_in_dim = int(config["enc_in_dim"])
        self.enc_dim = int(config["enc_dim"])
        self.pred_dim = int(config["pred_dim"])
        self.pred_layers = int(config["pred_layers"])
        self.eta_min = float(config["eta_min"])
        self.w_lin_mag = float(config["w_lin_mag"])
        self.w_log_mag = 1 - self.w_lin_mag
        self.fft_sizes = [int(i) for i in config["loss_ffts"]]
        self.disc_lr = CONFIG.DISCRIMINATOR.lr
        self.hop_sizes = [480, 960, 128] #[i // 4 for i in self.fft_sizes]
        self.disc_alpha = CONFIG.DISCRIMINATOR.adv_disc
        self.gen_alpha = CONFIG.DISCRIMINATOR.adv_gen

        self.stoi = STOI(CONFIG.DATA.sr)
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
        self.msd = MultiScaleDiscriminator()
        self.fm_loss = FMLoss(CONFIG.DISCRIMINATOR.fm_alpha)
        self.disc_loss = LeastSquaresDiscLoss()
        self.gen_loss = LeastSquaresGenLoss()
        self.loss = Loss(w_log_mag=self.w_log_mag, w_lin_mag=self.w_lin_mag,
                         fft_sizes=self.fft_sizes, hop_sizes=self.hop_sizes)
        self.window = torch.sqrt(torch.hann_window(self.window_size)).to(self.device)
        # self.save_hyperparameters('window_size', 'enc_layers', 'enc_in_dim', 'enc_dim', 'pred_dim', 'pred_layers')

    def forward(self, x, mask):
        #     """
        #     Input: real-imaginary; shape (B, 2, F, T); F = hop_size
        #            mask; shape (B, seq_len)
        #     Output: real-imaginary
        #     """
        B, C, F, T = x.shape
        x = x.permute(3, 0, 1, 2).unsqueeze(-1) #T B C F
        expanded_mask = mask.permute(1, 0).unsqueeze(2).repeat(1, 1, C).unsqueeze(3).repeat(1, 1, 1, F).unsqueeze(-1)
        prev_mag = torch.zeros((B, 1, F, 1), device=x.device)
        predictor_state = torch.zeros((2, self.predictor.lstm_layers, B, self.predictor.lstm_dim),
                                      device=x.device)
        mlp_state = torch.zeros((self.encoder.depth, 2, 1, B, self.encoder.dim), device=x.device)
        result = []
        for i in range(T):
            step = x[i].to(self.device)
            feat, mlp_state = self.encoder(step, mlp_state)
            prev_mag, predictor_state = self.predictor(prev_mag, predictor_state)
            feat = torch.cat((feat, prev_mag), dim=1)
            feat = self.joiner(feat)
            feat = feat + step
            # if packet is not lost use gt
            feat = torch.where(expanded_mask[i] > 0, x[i], feat)
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

    def training_step(self, batch, batch_idx, optimizer_idx):
        inp, tar, mask = batch
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        mask = mask.to(self.device)
        f_0 = inp[:, :, 0:1, :]
        x = inp[:, :, 1:, :]
        x = self(x, mask)
        x = torch.cat([f_0, x], dim=2)
        y = torch.view_as_complex(tar.permute(0, 2, 3, 1).contiguous())
        y = torch.istft(y, self.window_size, self.hop_size, window=self.window.to(self.device))
        pred = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        pred = torch.istft(pred, self.window_size, self.hop_size, window=self.window.to(self.device))
        if optimizer_idx == 0:
            disc_real_out, disc_gen_out = self.msd(y.unsqueeze(1)), self.msd(pred.unsqueeze(1))
            fm_loss = self.fm_loss(disc_real_out, disc_gen_out)
            gen_adv_loss = self.gen_alpha * self.gen_loss(disc_gen_out)
            loss = self.loss(x, tar)
            total_loss = loss + fm_loss + gen_adv_loss
            log_dict = {"train_stft_loss": loss.item(),
                        "train_fm_loss": fm_loss.item(),
                        "train_gen_adv_loss": gen_adv_loss.item(),
                        "optimizer_rate/optimizer": self.optimizer.param_groups[0]['lr']}
            self.log_dict(log_dict, on_step=True, on_epoch=False)
        else:
            disc_real_out, disc_gen_out = self.msd(y.unsqueeze(1)), self.msd(pred.unsqueeze(1).detach())
            disc_adv_loss = self.disc_alpha * self.disc_loss(disc_real_out, disc_gen_out)
            total_loss = disc_adv_loss
            log_dict = {"train_disc_adv_loss": disc_adv_loss.item(),
                        "optimizer_rate/optimizer_disc": self.disc_optimizer.param_groups[0]['lr']}
            self.log_dict(log_dict, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        inp, tar, mask = val_batch
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        mask = mask.to(self.device)
        f_0 = inp[:, :, 0:1, :]
        x_in = inp[:, :, 1:, :]
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
            "val_stft_loss": loss.item()
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        with torch.no_grad():
            inp, tar, mask, name = test_batch
            inp = inp.to(self.device)
            tar = tar.to(self.device)
            mask = mask.to(self.device)
            f_0 = inp[:, :, :1, :]
            x = inp[:, :, 1:, :]
            pred = self(x, mask)
            pred = torch.cat([f_0, pred], dim=2)
            loss = self.loss(pred, tar)
            self.window = self.window.to(pred.device)
            pred = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())
            pred = torch.istft(pred, self.window_size, self.hop_size, window=self.window)
            y = torch.view_as_complex(tar.permute(0, 2, 3, 1).contiguous())
            y = torch.istft(y, self.window_size, self.hop_size, window=self.window)
            loosy_signal = torch.view_as_complex(inp.permute(0, 2, 3, 1).contiguous())
            loosy_signal = torch.istft(loosy_signal, self.window_size, self.hop_size, window=self.window)
            for i in range(inp.shape[0]):
                y_i = y[i]
                pred_i = pred[i]
                out_dir_path = Path(CONFIG.NBTEST.out_dir) / name[i]
                out_dir_orig_path = Path(CONFIG.NBTEST.out_dir_orig) / name[i]
                sf.write(out_dir_orig_path, loosy_signal[i].squeeze(0).cpu().numpy(), samplerate=CONFIG.DATA.sr, subtype='PCM_16')
                sf.write(out_dir_path, pred_i.squeeze(0).cpu().numpy(), samplerate=CONFIG.DATA.sr, subtype='PCM_16')

            stoi = self.stoi(pred, y)
            stoi_predicted = self.stoi(loosy_signal, y)

            if CONFIG.DATA.sr != 16000:
                pred, y = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
                loosy_signal = loosy_signal.detach().cpu().numpy()
                pred = librosa.resample(pred, orig_sr=48000, target_sr=16000)
                y = librosa.resample(y, orig_sr=48000, target_sr=16000, res_type='kaiser_fast')
                loosy_signal = librosa.resample(loosy_signal, orig_sr=48000, target_sr=16000, res_type='kaiser_fast')

            intrusives = []
            non_intrusives = []
            lsds = []
            for i in range(pred.shape[0]):
                ret = plcmos.run(pred[i, :], y[i, :])
                intrusives.append(ret[0])
                non_intrusives.append(ret[1])
                lsds.append(LSD(y[i, :], pred[i, :])[0])

            loosy_intrusives = []
            loosy_non_intrusives = []
            loosy_lsds = []
            for i in range(loosy_signal.shape[0]):
                ret = plcmos.run(loosy_signal[i, :], y[i, :])
                loosy_intrusives.append(ret[0])
                loosy_non_intrusives.append(ret[1])
                loosy_lsds.append(LSD(y[i, :], loosy_signal[i, :])[0])

            log_dict = {
                "Intrusive": float(np.mean(intrusives)),
                "Non-intrusive": float(np.mean(non_intrusives)),
                'LSD': float(np.mean(lsds)),
                'STOI': stoi,
                "val_stft_loss": loss.item(),
                "Loosy Intrusive": float(np.mean(loosy_intrusives)),
                "Loosy Non-intrusive": float(np.mean(loosy_non_intrusives)),
                'Loosy LSD': float(np.mean(loosy_lsds)),
                'Loosy STOI': stoi_predicted,
            }
            print(log_dict)
            self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
            return log_dict

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
        self.optimizer = torch.optim.Adam(itertools.chain(self.predictor.parameters(),
                                                          self.joiner.parameters(),
                                                          self.encoder.parameters()),
                                          lr=self.learning_rate)
        self.disc_optimizer = torch.optim.Adam(self.msd.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=CONFIG.TRAIN.patience,
        #                                                           factor=CONFIG.TRAIN.factor, verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CONFIG.TRAIN.epochs,
                                                                  eta_min=1e-8, verbose=True)
        disc_scheduler = torch.optim.lr_scheduler.ConstantLR(self.disc_optimizer, factor=1)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=9e-1, last_epoch=CONFIG.TRAIN.epochs)

        # scheduler = {
        #     'scheduler': lr_scheduler,
        #     # 'reduce_on_plateau': True,
        #     'monitor': CONFIG.WANDB.monitor
        # }
        return [self.optimizer, self.disc_optimizer], [lr_scheduler, disc_scheduler]


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

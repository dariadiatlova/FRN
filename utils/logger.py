import librosa as rosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.stft import STFTMag

matplotlib.use('Agg')


class WandbSpectrogramLogging:
    def __init__(self):
        self.stftmag = STFTMag()

    def fig2np(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def plot_spectrogram_to_numpy(self, y, y_low, y_recon, step):
        name_list = ['y', 'y_low', 'y_recon']
        fig = plt.figure(figsize=(9, 15))
        fig.suptitle(f'Epoch_{step}')
        for i, yy in enumerate([y, y_low, y_recon]):
            if yy.dim() == 1:
                yy = self.stftmag(yy)
            ax = plt.subplot(3, 1, i + 1)
            ax.set_title(name_list[i])
            plt.imshow(rosa.amplitude_to_db(yy.numpy(), ref=np.max, top_db=80.),
                       vmax=0.,
                       aspect='auto',
                       origin='lower',
                       interpolation='none')
            plt.colorbar()
            plt.xlabel('Frames')
            plt.ylabel('Channels')
            plt.tight_layout()

        fig.canvas.draw()
        data = self.fig2np(fig)

        plt.close()
        return data


def plot_spectrogram_to_numpy(self, y, y_low, y_recon, step):
    name_list = ['y', 'y_low', 'y_recon']
    fig = plt.figure(figsize=(9, 15))
    fig.suptitle(f'Epoch_{step}')
    for i, yy in enumerate([y, y_low, y_recon]):
        if yy.dim() == 1:
            yy = self.stftmag(yy)
        ax = plt.subplot(3, 1, i + 1)
        ax.set_title(name_list[i])
        plt.imshow(rosa.amplitude_to_db(yy.numpy(),
                                        ref=np.max, top_db=80.),
                   # vmin = -20,
                   vmax=0.,
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
        plt.colorbar()
        plt.xlabel('Frames')
        plt.ylabel('Channels')
        plt.tight_layout()

    fig.canvas.draw()
    data = self.fig2np(fig)

    plt.close()
    return data

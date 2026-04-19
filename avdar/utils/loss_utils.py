import torch
import torch.nn as nn
import torch.linalg

import numpy as np

import scipy.signal as signal

import auraloss

from typing import List

torch.set_default_dtype(torch.float32)

def normalized(x: torch.Tensor, dim = None, eps = 1e-12):
    """
    Nomalize a tensor with L2 norm
    """
    
    norm = torch.linalg.norm(x, dim=dim)
    
    if dim is None:
        return x if norm < eps else x / norm
    
    output = torch.clone(x)
    output[norm > eps] = (x / torch.unsqueeze(norm, dim)) [norm > eps]
    return output




def safe_log(x, eps=1e-7):
    """
    Avoid taking the log of a non-positive number
    """
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)


def get_stft(x, n_fft, hop_length=None):
    """
    Returns the stft of x.
    """
    return torch.stft(x,
                      n_fft=n_fft,
                      hop_length = hop_length,
                      window=torch.hann_window(n_fft).to(x.device),
                      return_complex=False)




"""
Training Losses
"""
def L1_and_Log(x,y, n_fft=512, hop_length=None, eps=1e-6):
    """
    Computes spectral L1 plus log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    n_fft: n_fft for stft
    hop_length: stft hop length
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss (float)
    """
    est_stft = get_stft(x, n_fft=n_fft,hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft,hop_length=hop_length)
    
    assert est_stft.shape == ref_stft.shape    
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)

    result = torch.mean(torch.abs(safe_log(est_amp)-safe_log(ref_amp))) + torch.mean(torch.abs(est_amp-ref_amp))
    # result = torch.mean(torch.abs(est_amp-ref_amp))
    return result

def L1_and_Log_Decay(x, y, n_fft=512, hop_length=None, eps=1e-6):
    est_stft = get_stft(x, n_fft=n_fft,hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft,hop_length=hop_length)

    assert est_stft.shape == ref_stft.shape

    est_energy = (est_stft[..., 0]**2 + est_stft[..., 1]**2).sum(dim=-2)
    ref_energy = (ref_stft[..., 0]**2 + ref_stft[..., 1]**2).sum(dim=-2)

    est_cum_energy = torch.cumsum(est_energy.flip(-1), dim=-1).flip(-1)
    ref_cum_energy = torch.cumsum(ref_energy.flip(-1), dim=-1).flip(-1)

    est_log_decay = torch.log(est_cum_energy[..., :-1]) - torch.log(est_cum_energy[..., 1:] + eps)
    ref_log_decay = torch.log(ref_cum_energy[..., :-1]) - torch.log(ref_cum_energy[..., 1:] + eps)

    result = torch.mean(torch.abs(est_log_decay - ref_log_decay))
    return result

    
def spec_loss(X, Y):
    D = torch.view_as_real(X) - torch.view_as_real(Y)
    return torch.mean(torch.abs(D[..., 0])) + torch.mean(torch.abs(D[..., 1]))

def phase_loss(X, Y):
    phase_X = torch.cos(torch.angle(X))
    phase_Y = torch.sin(torch.angle(Y))
    return torch.mean(torch.abs(phase_X - phase_Y))

def amplitude_loss(X, Y):
    amp_X = torch.abs(X)
    amp_Y = torch.abs(Y)
    return torch.mean(torch.abs(amp_X - amp_Y))

def time_loss(x, y):
    return torch.mean(torch.abs(x - y))

def training_loss(x,y,cutoff=9000, eps=1e-6):
    """
    Training Loss

    Computes spectral L1 and log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss: float tensor
    """
    loss1 = L1_and_Log(x,y, n_fft=512, eps=eps)
    loss2 = L1_and_Log(x,y, n_fft=1024, eps=eps)
    # loss3 = loss4 = 0
    loss3 = L1_and_Log(x,y, n_fft=2048, eps=eps)
    loss4 = L1_and_Log(x,y, n_fft=4096, eps=eps) # regular in diffrir
    # loss4 = L1_and_Log(x,y, n_fft=256, eps=eps)
    # loss4 = 0
    tiny_hop_loss = L1_and_Log(x[...,:cutoff], y[...,:cutoff], n_fft=256, eps=eps, hop_length=1)
    return loss1 + loss2 + loss3 + loss4 + tiny_hop_loss
    

def decay_loss(x, y, eps=1e-6):
    
    loss4 = L1_and_Log_Decay(x,y, n_fft=128, eps=eps)
    
    loss1 = L1_and_Log_Decay(x,y, n_fft=512, eps=eps)
    loss2 = L1_and_Log_Decay(x,y, n_fft=1024, eps=eps)
    loss3 = L1_and_Log_Decay(x,y, n_fft=2048, eps=eps)
    # loss4 = L1_and_Log_Decay(x,y, n_fft=4096, eps=eps)
    # loss4 = 0
    return loss1 + loss2 + loss3 + loss4

"""
Evaluation Metrics
"""

def log_L1_STFT(x,y, n_fft=512, eps=1e-6, hop_length=None):
    """
    Computes log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    n_fft: n_fft for stft
    hop_length: stft hop length
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss, float tensor
    """
    est_stft = get_stft(x, n_fft=n_fft, hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft, hop_length=hop_length)
    
    assert est_stft.shape == ref_stft.shape 

    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)
    result = torch.mean(torch.abs(safe_log(est_amp)-safe_log(ref_amp)))

    return result

def multiscale_log_l1(x,y, eps=1e-6):
    """Spectral Evaluation Metric"""
    loss = 0
    loss += log_L1_STFT(x,y, n_fft=64, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=128, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=256, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=512, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=1024, eps=eps)
    if x.shape[-1] > 2048:
        loss += log_L1_STFT(x,y, n_fft=2048, eps=eps)
    if x.shape[-1] > 4096:
        loss += log_L1_STFT(x,y, n_fft=4096, eps=eps)
    # loss += log_L1_STFT(x,y, n_fft=2048, eps=eps)
    # loss += log_L1_STFT(x,y, n_fft=4096, eps=eps)
    return loss

def env_loss(x, y, envelope_size=32, eps=1e-6):
    """Envelope Evaluation Metric. x,y are tensors representing waveforms."""
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    env1 = signal.convolve(x**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps
    env2 = signal.convolve(y**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps

    loss =  (np.mean(np.abs(np.log(env1) - np.log(env2))))
    
    return loss

baseline_metrics = [multiscale_log_l1, env_loss]

def LRE(x, y, n_fft = 1024, hop_length=None, eps=1e-6):
    """LRE - Binaural Evaluation."""
    est_stft = get_stft(x, n_fft=n_fft, hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft, hop_length=hop_length)

    assert est_stft.shape == ref_stft.shape    
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)
    dif = torch.sum(est_amp[1])/torch.sum(est_amp[0]) - torch.sum(ref_amp[1])/torch.sum(ref_amp[0])
    dif = dif ** 2

    return dif.item()

def measure_edt_inras(h, fs=22050, decay_db=10):
    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    if (energy > 0).sum() == 0:
        return 100

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs
    # compute the decay time
    decay_time = t_decay
    est_edt = (60 / decay_db) * decay_time
    return est_edt

def measure_rt60_inras(h, fs=22050, decay_db=30):
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.
    Parameters
    ----------
    h: array_like
        The impulse response.
    fs: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    decay_db: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    plot: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    rt60_tgt: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    """

    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder
    
    if energy.max() <= 0:
        return 100
    
    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]
    # -5 dB headroom
    i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs

    # after decay
    if len(np.where(-5-decay_db - energy_db >0)[0]) == 0:
        return 100
    i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time
    # c50 = 10.0 * np.log10((np.sum(pow_energy[:t]) / np.sum(pow_energy[t:])))

    return est_rt60


def measure_c50_inras(h, fs=22050):
    h = np.array(h)
    fs = float(fs)

    trunc = np.round(0.05 * fs).astype(int)
    # The power of the impulse response in dB
    power = h ** 2

    c50 = 10.0 * np.log10((np.sum(power[:trunc]) / (np.sum(power[trunc:]) + 1e-12)) + 1e-12)
    return c50

def measure_rt60(h, fs=48000, decay_db=30):
    power = np.array(h) ** 2 + 1e-9
    energy = np.cumsum(power[::-1])[::-1]
    rel_energy_dB = 10 * np.log10(energy / energy[0])
    decay_5dB_index = np.sum(rel_energy_dB > -5 - 1e-5)
    decay_65dB_index = np.sum(rel_energy_dB > -5 - decay_db - 1e-5)
    rt60 = 60 / decay_db * (decay_65dB_index - decay_5dB_index) / fs
    return rt60

def measure_c50(h, fs=48000):
    stop_index = int(50 / 1000 * fs + .5)

    power = np.array(h) ** 2 + 1e-9
    energy = np.cumsum(power[::-1])[::-1]

    if energy[stop_index] == 0:
        return 0
    
    tot_energy = energy[0]
    stop_energy = energy[stop_index]

    rel_energy_decay_dB = 10 * np.log10((tot_energy - stop_energy) / stop_energy)

    return rel_energy_decay_dB

def measure_edt(h, fs=48000, decay_db=10):
    power = np.array(h) ** 2 + 1e-9
    energy = np.cumsum(power[::-1])[::-1]

    rel_energy_dB = 10 * np.log10(energy / energy[0])
    start_decay_index = np.sum(rel_energy_dB > -1e-5)
    decay_10dB_index = np.sum(rel_energy_dB > -decay_db - 1e-5)
    edt = (decay_10dB_index - start_decay_index) / fs * 60 / decay_db

    return edt

def measure_energy(h, fs=48000):
    power = np.array(h) ** 2
    int_energy = np.sum(power) / fs # riemann sum
    int_log_energy = 10 * np.log10(int_energy + 1e-9)
    return int_log_energy

def raf_stft_error(x, y):
    ...

def raf_c50_error(x, y):
    ...

class RafEdtError:
    def __init__(self, fs, decay_db = 10):
        self.fs = fs
        self.decay_db = decay_db

    def __call__(self, x, y):
        edt_x = measure_edt(x, fs=self.fs, decay_db=self.decay_db)
        edt_y = measure_edt(y, fs=self.fs, decay_db=self.decay_db)
        error = torch.abs(torch.tensor(edt_x - edt_y))
        return error

# y gt, x pred
class RafT60Error:
    def __init__(self, fs, decay_db = 20):
        self.fs = fs
        self.decay_db = decay_db

    def __call__(self, x, y):
        t60_x = measure_rt60(x, fs=self.fs, decay_db=self.decay_db)
        t60_y = measure_rt60(y, fs=self.fs, decay_db=self.decay_db)
        error = torch.abs(torch.tensor((t60_x - t60_y) / t60_y))
        return error
    
class RafC50Error:
    def __init__(self, fs):
        self.fs = fs

    def __call__(self, x, y):
        c50_x = measure_c50(x, fs=self.fs)
        c50_y = measure_c50(y, fs=self.fs)
        error = torch.abs(torch.tensor(c50_x - c50_y))
        return error
    
class StftError:
    def __init__(self):
        self.multi_stft = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, w_sc=0)

    def __call__(self, x, y):
        return self.multi_stft(x[None, None], y[None, None])
    
class PhaseError:
    def __init__(self):
        ...

    def __call__(self, x, y):
        x = torch.fft.fft(x, dim=-1)
        y = torch.fft.fft(y, dim=-1)
        return torch.mean(torch.abs(torch.cos(torch.angle(x)) - torch.cos(torch.angle(y)))) + \
                torch.mean(torch.abs(torch.sin(torch.angle(x)) - torch.sin(torch.angle(y))))
    
class AmplitudeError:
    def __init__(self, window=32):
        self.window = window

    def __call__(self, x, y):
        x = torch.fft.fft(x, dim=-1)
        y = torch.fft.fft(y, dim=-1)
        x_amp = torch.abs(x)
        y_amp = torch.abs(y)
        x_amp_conv = torch.nn.functional.conv1d(x_amp[None, None], torch.ones(1, 1, self.window).to(x.device), padding=self.window//2)[0, 0]
        y_amp_conv = torch.nn.functional.conv1d(y_amp[None, None], torch.ones(1, 1, self.window).to(y.device), padding=self.window//2)[0, 0]
        return torch.mean(torch.abs(x_amp_conv - y_amp_conv) / (y_amp_conv + 1e-9))
    
class EnvError:
    def __init__(self):
        ...

    def __call__(self, x, y):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        x_env = np.abs(signal.hilbert(x))
        y_env = np.abs(signal.hilbert(y))
        return np.mean(np.abs(x_env - y_env) / (np.max(y_env, axis=-1, keepdims=True) + 1e-9))

    
class RafLoudnessError:
    def __init__(self, fs):
        self.fs = fs

    def __call__(self, x, y):
        energy_x = measure_energy(x, fs=self.fs)
        energy_y = measure_energy(y, fs=self.fs)
        error = torch.abs(torch.tensor(energy_x - energy_y))
        return error

class DiffRirLoss(nn.Module):
    def __init__(self, 
                 nffts: List[int] = [128, 512, 1024, 2048],
                 cutoff: int = 3000,
                 eps: float = 1e-6):
        super().__init__()
        self.nffts = nffts
        self.eps = eps
        self.cutoff = cutoff

    def forward(self, x, y):
        loss = 0
        for n_fft in self.nffts:
            loss += L1_and_Log(x, y, n_fft=n_fft, eps=self.eps)
        tiny_hop_loss = L1_and_Log(x[...,:self.cutoff], y[...,:self.cutoff], n_fft=128, eps=self.eps, hop_length=1)
        return loss + tiny_hop_loss

class AvrLoss(nn.Module):
    def __init__(self, lambda_spec_loss, lambda_amplitude_loss, lambda_angle_loss, lambda_time_loss, lambda_energy_loss, lambda_multi_stft_loss, 
                 mrft_loss_options = {
                        'w_lin_mag': 1.0,
                        'fft_sizes': [2048, 1024, 512, 256, 128, 64],
                        'win_lengths': [1200, 600, 300, 150, 75, 30],
                        'hop_sizes': [240, 120, 60, 30, 8, 4]
                 }):
        super().__init__()

        self.lambda_spec_loss = lambda_spec_loss
        self.lambda_amplitude_loss = lambda_amplitude_loss
        self.lambda_angle_loss = lambda_angle_loss
        self.lambda_time_loss = lambda_time_loss
        self.lambda_energy_loss = lambda_energy_loss
        self.lambda_multi_stft_loss = lambda_multi_stft_loss

        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.mrft_loss = auraloss.freq.MultiResolutionSTFTLoss(**mrft_loss_options)

    def forward(self, pred_time, ori_time):

        if len(pred_time.shape) == 1:
            pred_time = pred_time.unsqueeze(0)
            ori_time = ori_time.unsqueeze(0)

        # pred_time = torch.real(torch.fft.irfft(pred_sig, dim=-1)) 
        # ori_time = torch.real(torch.fft.irfft(ori_sig, dim=-1))
        pred_sig = torch.fft.rfft(pred_time, dim=-1)
        ori_sig = torch.fft.rfft(ori_time, dim=-1)

        pred_spec = torch.abs(torch.stft(pred_time, n_fft=256, return_complex=True))
        ori_spec = torch.abs(torch.stft(ori_time, n_fft=256, return_complex=True))

        pred_spec_energy = torch.sum(pred_spec ** 2, dim=1)
        ori_spec_energy = torch.sum(ori_spec ** 2, dim=1)

        predict_energy = torch.log10(torch.flip(torch.cumsum(torch.flip(pred_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        predict_energy -= predict_energy[:,[0]]
        ori_energy = torch.log10(torch.flip(torch.cumsum(torch.flip(ori_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        ori_energy -= ori_energy[:,[0]]

        real_loss = self.l1_loss(torch.real(pred_sig), torch.real(ori_sig))
        imag_loss  = self.l1_loss(torch.imag(pred_sig), torch.imag(ori_sig)) 
        spec_loss = (real_loss + imag_loss) * self.lambda_spec_loss

        amplitude_loss = self.l1_loss(torch.abs(pred_sig), torch.abs(ori_sig)) * self.lambda_amplitude_loss 

        angle_loss = (self.l1_loss(torch.cos(torch.angle(pred_sig)), torch.cos(torch.angle(ori_sig))) + \
                    self.l1_loss(torch.sin(torch.angle(pred_sig)), torch.sin(torch.angle(ori_sig)))) * self.lambda_angle_loss
        
        time_loss = self.l1_loss(ori_time, pred_time) * self.lambda_time_loss

        energy_loss = self.l1_loss(ori_energy, predict_energy) * self.lambda_energy_loss

        multi_stft_loss = self.mrft_loss(ori_time.unsqueeze(1), pred_time.unsqueeze(1)) * self.lambda_multi_stft_loss

        

        # return spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, ori_time, pred_time

        return {
            'spec_loss': spec_loss,
            'amplitude_loss': amplitude_loss,
            'angle_loss': angle_loss,
            'time_loss': time_loss,
            'energy_loss': energy_loss,
            'multi_stft_loss': multi_stft_loss
        }
    

class MixedAvrLoss(nn.Module):
    def __init__(self, lambda_spec_loss, lambda_amplitude_loss, lambda_angle_loss, lambda_time_loss, lambda_energy_loss, lambda_multi_stft_loss, lambda_diff_rir_loss):

        super().__init__()

        self.avr_loss = AvrLoss(lambda_spec_loss, lambda_amplitude_loss, lambda_angle_loss, lambda_time_loss, lambda_energy_loss, lambda_multi_stft_loss)

        self.lambda_diff_rir_loss = lambda_diff_rir_loss

    def forward(self, pred_time, ori_time):
        
        avr_loss = self.avr_loss(pred_time, ori_time)
        diff_rir_loss = training_loss(pred_time, ori_time)

        return {
            **avr_loss,
            'diff_rir_loss': diff_rir_loss
        }
    
PRESET_ACOUSTIC_METRICS = {
    'C50': measure_c50,
    'EDT': measure_edt,
    'T60': measure_rt60,
    'T60 (by T20)': lambda h, fs: measure_rt60(h, fs=fs, decay_db=20),
}
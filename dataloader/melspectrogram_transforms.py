import torch
import torchaudio
import torchaudio.transforms as T
import yaml

# yaml config
with open('config_melspec.yaml', 'r') as file:
    params = yaml.safe_load(file)

def melspectrogram_transforms(samples, params):
    """
    :param samples: audio samples -> torch
    :return: mel spectrogram (n_frames, n_mels (fre bins))
    """
    params["window_fn"] = torch.hamming_window
    F_mel = T.MelSpectrogram(
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        win_length=params["win_length"],
        hop_length=params["hop_length"],
        window_fn=params["window_fn"],
        center=params["center"],
        pad_mode=params["pad_mode"],
        power=params["power"],
        norm=params["norm"],
        n_mels=params["n_mels"],
        mel_scale=params["mel_scale"]
    )
    # mel spectrogram transforms
    mel_spec = F_mel(samples)
    result = torch.squeeze(mel_spec, 0)
    # return n_mels, n_frames
    return result
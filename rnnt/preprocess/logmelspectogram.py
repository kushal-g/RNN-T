from config.config import speech_config
import os
import librosa
import numpy as np
import tensorflow as tf

def _normalize_signal( signal: np.ndarray,) :
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
    return signal * gain

def _preemphasis( signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def _load_file( path: str):
    wave, _ = librosa.load(os.path.expanduser(path),sr=speech_config["sample_rate"], mono=True)
    return wave

def _stft(signal):
  frame_length = int(speech_config["sample_rate"] * (speech_config["frame_ms"] / 1000))
  frame_step = int(speech_config["sample_rate"] * (speech_config["stride_ms"] / 1000))
  nfft = 2 ** (frame_length - 1).bit_length()
  if speech_config["center"]:
        signal = tf.pad(signal, [[nfft // 2, nfft // 2]], mode="REFLECT")
  window = tf.signal.hann_window(frame_length, periodic=True)
  left_pad = (nfft - frame_length) // 2
  right_pad = nfft - frame_length - left_pad
  window = tf.pad(window, [[left_pad, right_pad]])
  framed_signals = tf.signal.frame(signal, frame_length=nfft, frame_step=frame_step)
  framed_signals *= window
  return tf.square(tf.abs(tf.signal.rfft(framed_signals, [nfft])))

def _compute_log_mel_spectrogram(signal):
        spectrogram = _stft(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=speech_config["feature_bins"],
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=speech_config["sample_rate"],
            lower_edge_hertz=0.0,
            upper_edge_hertz=(speech_config["sample_rate"] / 2),
        )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
        return librosa.power_to_db(mel_spectrogram)

def get(path: str):
    signal = _load_file(path)
    if speech_config["normalize"]:
        signal = _normalize_signal(signal)
    
    signal = _preemphasis(signal,coeff=speech_config["preemphasis"])
    return _compute_log_mel_spectrogram(signal)

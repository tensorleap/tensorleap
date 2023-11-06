import numpy as np
from code_loader.contract.visualizer_classes import LeapText, LeapImage, LeapGraph
import numpy.typing as npt

from librispeech_clean.configuration import config
from librispeech_clean.utils import remove_trailing_zeros, normalize_array
from librispeech_clean.wav2vec_processor import ProcessorSingleton
from librosa.feature import melspectrogram, rms
from librosa import power_to_db, resample

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx

cmap = plt.get_cmap('magma')
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)


def display_predicted_transcription(data: npt.NDArray[np.float32]) -> LeapText:
    processor = ProcessorSingleton().get_processor()
    predicted_ids = np.argmax(data, axis=1)
    text = [processor.decode(predicted_ids)]
    return LeapText(text)


def display_gt_transcription(data: npt.NDArray[np.float32]) -> LeapText:
    numeric_labels = remove_trailing_zeros(data)
    processor = ProcessorSingleton().get_processor()
    text = [processor.tokenizer.decode(numeric_labels)]
    return LeapText(text)


def display_mel_spectrogram(data: npt.NDArray[np.float32]) -> LeapImage:
    # data_trimmed = remove_trailing_zeros(data)
    resized_data = data[:config.get_parameter('clip_visualizers')]
    ms = melspectrogram(y=resized_data, sr=config.get_parameter('sampling_rate'), )
    ms_db = power_to_db(ms, ref=np.max)
    scaled_ms = normalize_array(ms_db)
    colored_depth = scalarMap.to_rgba(scaled_ms)[..., :-1]
    res = colored_depth.astype(np.float32) * 255.0
    return LeapImage(res)


def display_mel_spectrogram_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    # data_trimmed = remove_trailing_zeros(data)
    data = np.squeeze(data)
    resized_data = data[:config.get_parameter('clip_visualizers')]
    agg_data = np.tile(rms(y=resized_data), [128, 1])
    res = np.expand_dims(agg_data, -1)
    return res


def display_waveform(data: npt.NDArray[np.float32]) -> LeapGraph:
    # data_trimmed = remove_trailing_zeros(data)
    resized_data = data[:config.get_parameter('clip_visualizers')]
    resized_data = resized_data.reshape(-1, 10).mean(1)
    # down_sampled_data = resample(resized_data,
    #                              orig_sr=config.get_parameter('sampling_rate'),
    #                              target_sr=config.get_parameter('sampling_rate') // 10)
    res = resized_data.reshape(-1, 1)
    return LeapGraph(res)


def display_waveform_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    # data_trimmed = remove_trailing_zeros(data)
    resized_data = data[:config.get_parameter('clip_visualizers')]
    resized_data = resized_data.reshape(-1, 10).mean(1)
    # down_sampled_data = resample(resized_data,
    #                              orig_sr=config.get_parameter('sampling_rate'),
    #                              target_sr=config.get_parameter('sampling_rate') // 10)
    res = resized_data.reshape(-1, 1)
    return res

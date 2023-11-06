from librispeech_clean.packages import install_all_packages

install_all_packages()

import pandas as pd
from typing import List, Dict, Union
import numpy as np
from code_loader.contract.enums import LeapDataType

from librispeech_clean.configuration import config
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse

from librispeech_clean.gcs_utils import download
from librispeech_clean.metrics import ctc_loss, calculate_error_rate_metrics
from librispeech_clean.utils import pad_gt_numeric_labels, get_speech_pauses_to_word_gaps
from librispeech_clean.visualizers import display_predicted_transcription, display_gt_transcription, display_mel_spectrogram, \
    display_mel_spectrogram_heatmap, display_waveform, display_waveform_heatmap
from librispeech_clean.wav2vec_processor import ProcessorSingleton
from librosa.feature import spectral_flatness, spectral_contrast


# -data processing-
def get_data_subsets() -> List[PreprocessResponse]:
    responses = []
    for dataset_slice, slice_dict in config.get_parameter('dataset_slices').items():
        if slice_dict['path'] is not None:
            fpath = download(slice_dict['path'])
            data = pd.read_csv(fpath, index_col=0)
            data = data.sample(n=slice_dict['n_samples'], random_state=config.get_parameter('seed'))
            response = PreprocessResponse(length=slice_dict['n_samples'], data=data)
            responses.append(response)
    return responses


def get_input_audio(idx: int, data: PreprocessResponse, padded: bool = True) -> np.ndarray:
    data = data.data
    audio_gcs_path = data.iloc[idx]['audio_path']
    fpath = download(audio_gcs_path)
    audio_array = np.load(fpath)[0]
    if not padded:
        return audio_array

    padding = config.get_parameter('max_sequence_length') - audio_array.size
    padded_audio_array = np.pad(audio_array, (0, padding))
    return padded_audio_array


def get_gt_transcription(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    processor = ProcessorSingleton().get_processor()
    transcription = data.iloc[idx]['text']
    numeric_labels = processor.tokenizer.encode(transcription)
    padded_labels = pad_gt_numeric_labels(numeric_labels)
    return padded_labels


# -metadata-
def get_metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[int, float, str]]:
    sample = data.data.iloc[idx]
    audio_array = get_input_audio(idx, data, padded=False)
    sf = spectral_flatness(y=audio_array)
    sc = spectral_contrast(y=audio_array)
    metadata = {
        'index': idx,
        'speaker_id': int(sample['speaker_id']),
        'chapter_id': int(sample['chapter_id']),
        'word_count': len(sample['text'].split()),
        'pauses_word_gaps_diff': get_speech_pauses_to_word_gaps(audio_array, sample['text']),
        'signal_mean': float(audio_array.mean()),
        'signal_std': float(audio_array.std()),
        'spectral_flatness_mean': float(sf.mean()),
        'spectral_flatness_std': float(sf.std()),
        'spectral_flatness_max': float(sf.max()),
        'spectral_flatness_min': float(sf.min()),
        'spectral_contrast_mean': float(sc.mean()),
        'spectral_contrast_std': float(sc.std()),
        'spectral_contrast_max': float(sc.max()),
        'spectral_contrast_min': float(sc.min())
    }
    metadata = {k: round(v, 3) for k, v in metadata.items()}
    return metadata


leap_binder.set_preprocess(get_data_subsets)
leap_binder.set_input(get_input_audio, 'audio_array')
leap_binder.set_ground_truth(get_gt_transcription, 'numeric_labels')
leap_binder.add_prediction('characters',
                           ['<pad>', '<s>', '</s>', '<unk>', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L',
                            'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z'])

leap_binder.add_custom_metric(calculate_error_rate_metrics, 'error_rate_metrics')
leap_binder.add_custom_loss(ctc_loss, 'ctc_loss')

leap_binder.set_metadata(get_metadata_dict, '')

leap_binder.set_visualizer(display_predicted_transcription, name='transcription',
                           visualizer_type=LeapDataType.Text)
leap_binder.set_visualizer(display_gt_transcription, name='reference',
                           visualizer_type=LeapDataType.Text)
leap_binder.set_visualizer(display_mel_spectrogram, name='mel_spectrogram',
                           heatmap_visualizer=display_mel_spectrogram_heatmap,
                           visualizer_type=LeapDataType.Image)
leap_binder.set_visualizer(display_waveform, name='waveform',
                           heatmap_visualizer=display_waveform_heatmap,
                           visualizer_type=LeapDataType.Graph)

if __name__ == '__main__':
    leap_binder.check()

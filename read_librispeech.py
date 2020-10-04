import os
import numpy as np

import soundfile as sf
import librosa

import torch

from data_utils import load_audio, FeatureNormalizer

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, directory, filter_gender=None, speaker_info_file=None):
        if filter_gender is not None:
            assert filter_gender in ['M','F']
            assert speaker_info_file is not None, 'must have speaker info file to filter gender'
            gender_map = {}
            with open(speaker_info_file, 'r') as f:
                for line in f:
                    if line.startswith(';'):
                        continue
                    line = line.split('|')
                    speaker_id = line[0].strip()
                    speaker_gender = line[1].strip()
                    gender_map[speaker_id] = speaker_gender

        self.base_directory = directory
        self.filenames = [] # tuples of (filename, speaker_index)
        speaker_index = 0
        for s in os.listdir(directory):
            if filter_gender is not None and gender_map[s] != filter_gender:
                continue
            subdirectory = os.path.join(directory, s)
            for f in os.listdir(subdirectory):
                filename = os.path.join(s, f)
                self.filenames.append((filename, speaker_index))
            speaker_index += 1

        self.num_speakers = speaker_index

        mfcc_samples = []
        for fn, s in self.filenames[:50]:
            mfccs, _ = load_audio(os.path.join(self.base_directory,fn))
            mfcc_samples.append(mfccs)
        self.mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)

        self.num_speech_features = mfcc_samples[0].shape[1]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn, s = self.filenames[idx]
        mfccs, audio_discrete = load_audio(os.path.join(self.base_directory, fn))
        mfccs = self.mfcc_norm.normalize(mfccs)
        return {'audio_features':mfccs, 'quantized_audio':audio_discrete, 'file_label':fn, 'speaker':s}


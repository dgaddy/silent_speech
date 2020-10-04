import re
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import scipy
import json
import copy
import sys
import pickle

import librosa
import soundfile as sf

import torch

from data_utils import load_audio, get_emg_features, FeatureNormalizer, split_fixed_length

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('remove_channels', [], 'channels to remove')
flags.DEFINE_list('silent_data_directories', [], 'silent data locations')
flags.DEFINE_list('audio_feature_directories', [], 'audio feature directories')
flags.DEFINE_list('voiced_data_directories', [], 'voiced data locations')
flags.DEFINE_string('testset_file', 'testset.json', 'file with testset indices')
flags.DEFINE_string('normalizers_file', 'normalizers.pkl', 'file with pickled feature normalizers')

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def load_utterance(base_dir, index, limit_length=False, debug=False):
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    before = os.path.join(base_dir, f'{index-1}_emg.npy')
    after = os.path.join(base_dir, f'{index+1}_emg.npy')
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    emg = []
    remove_channels = [int(c) for c in FLAGS.remove_channels]
    for emg_channel in range(raw_emg.shape[1]):
        # add more context for filtering, then remove after filtering
        x = np.concatenate([raw_emg_before[:,emg_channel],
                            raw_emg[:,emg_channel],
                            raw_emg_after[:,emg_channel]])
        x = notch_harmonics(x, 60, 1000)
        x = remove_drift(x, 1000)
        x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0]]
        x = subsample(x, 600, 1000)

        if emg_channel in remove_channels:
            x = np.zeros_like(x)

        emg.append(x)

        if debug:
            plt.plot(x)
            plt.show()
            s = abs(librosa.stft(np.ascontiguousarray(x)))
            plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')
            plt.show()

    emg = np.stack(emg, 1)
    emg_features = get_emg_features(emg)

    mfccs, audio_discrete = load_audio(os.path.join(base_dir, f'{index}_audio_clean.flac'),
            max_frames=min(emg_features.shape[0], 800 if limit_length else float('inf')))

    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[:mfccs.shape[0],:]

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    return mfccs, audio_discrete, emg_features, info['text'], (info['book'],info['sentence_index'])

class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory

class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, limit_length=False, dev=False, test=False, no_testset=False, no_normalizers=False):
        if no_testset:
            devset = []
            testset = []
        else:
            with open(FLAGS.testset_file) as f:
                testset_json = json.load(f)
                devset = testset_json['dev']
                testset = testset_json['test']

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            for sd in FLAGS.silent_data_directories:
                for session_dir in sorted(os.listdir(sd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))

            has_silent = len(FLAGS.silent_data_directories) > 0
            for vd in FLAGS.voiced_data_directories:
                for session_dir in sorted(os.listdir(vd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False, exclude_from_testset=has_silent))

        self.example_indices = []
        self.voiced_data_locations = {} # map from book/sentence_index to directory_info/index
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.json', fname)
                if m is not None:
                    idx_str = m.group(1)
                    with open(os.path.join(directory_info.directory, fname)) as f:
                        info = json.load(f)
                        if info['sentence_index'] >= 0: # boundary clips of silence are marked -1
                            location_in_testset = [info['book'], info['sentence_index']] in testset
                            location_in_devset = [info['book'], info['sentence_index']] in devset
                            if (test and location_in_testset and not directory_info.exclude_from_testset) \
                                    or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                                    or (not test and not dev and not location_in_testset and not location_in_devset):
                                self.example_indices.append((directory_info,int(idx_str)))

                            if not directory_info.silent:
                                location = (info['book'], info['sentence_index'])
                                self.voiced_data_locations[location] = (directory_info,int(idx_str))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.mfcc_norm, self.emg_norm = pickle.load(open(FLAGS.normalizers_file,'rb'))

        sample_mfccs, _, sample_emg, _, _ = load_utterance(self.example_indices[0][0].directory, self.example_indices[0][1])
        self.num_speech_features = sample_mfccs.shape[1]
        self.num_features = sample_emg.shape[1]
        self.limit_length = limit_length
        self.num_sessions = len(directories)

        self.alignments = None

    def silent_subset(self):
        silent_indices = []
        for i, (d, _) in enumerate(self.example_indices):
            if d.silent:
                silent_indices.append(i)
        return torch.utils.data.Subset(self, silent_indices)

    def set_silent_alignments(self, examples, alignments):
        self.alignments = {ex['book_location']:al for ex, al in zip(examples, alignments)}

    def __len__(self):
        return len(self.example_indices)

    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        mfccs, audio, emg, text, book_location = load_utterance(directory_info.directory, idx, self.limit_length)

        if not self.no_normalizers:
            mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg)
            emg = 8*np.tanh(emg/8.)

        session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)

        result = {'audio_features':mfccs, 'quantized_audio':audio, 'emg':emg, 'text':text, 'file_label':idx, 'session_ids':session_ids, 'book_location':book_location, 'silent':directory_info.silent}

        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, _, voiced_emg, _, _ = load_utterance(voiced_directory.directory, voiced_idx, False)

            if not self.no_normalizers:
                voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
                voiced_emg = self.emg_norm.normalize(voiced_emg)
                voiced_emg = 8*np.tanh(voiced_emg/8.)

            result['parallel_voiced_audio_features'] = voiced_mfccs
            result['parallel_voiced_emg'] = voiced_emg

            if self.alignments is not None:
                alignment = self.alignments[book_location]
                result['audio_features'] = voiced_mfccs[alignment]

        return result

    @staticmethod
    def collate_fixed_length(batch):
        batch_size = len(batch)
        audio_features = torch.cat([torch.from_numpy(ex['audio_features']) for ex in batch], 0)
        emg = torch.cat([torch.from_numpy(ex['emg']) for ex in batch], 0)
        quantized_audio = torch.cat([torch.from_numpy(ex['quantized_audio']) for ex in batch], 0)
        session_ids = torch.cat([torch.from_numpy(ex['session_ids']) for ex in batch], 0)

        mb = batch_size*8
        return {'audio_features':split_fixed_length(audio_features, 100)[:mb],
                'emg':split_fixed_length(emg, 100)[:mb],
                'quantized_audio':split_fixed_length(quantized_audio, 16000)[:mb],
                'session_ids':split_fixed_length(session_ids, 100)[:mb]}

def make_normalizers():
    dataset = EMGDataset(no_normalizers=True)
    mfcc_samples = []
    emg_samples = []
    for d in dataset:
        mfcc_samples.append(d['audio_features'])
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump((mfcc_norm, emg_norm), open(FLAGS.normalizers_file, 'wb'))

if __name__ == '__main__':
    FLAGS(sys.argv)
    d = EMGDataset()
    emg_features = d[0]['emg']
    plt.plot(emg_features[:,:8])
    plt.show()


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
import string
import logging

import librosa
import soundfile as sf
from textgrids import TextGrid

import torch

from data_utils import load_audio, get_emg_features, FeatureNormalizer, combine_fixed_length, phoneme_inventory

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('remove_channels', [], 'channels to remove')
flags.DEFINE_list('silent_data_directories', ['./emg_data/silent_parallel_data'], 'silent data locations')
flags.DEFINE_list('voiced_data_directories', ['./emg_data/voiced_parallel_data','./emg_data/nonparallel_data'], 'voiced data locations')
flags.DEFINE_string('testset_file', 'testset_largedev.json', 'file with testset indices')
flags.DEFINE_string('text_align_directory', 'text_alignments', 'directory with alignment files')

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

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)

def load_utterance(base_dir, index, limit_length=False, debug=False, text_align_directory=None):
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

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]
    emg_orig = apply_to_all(subsample, x, 800, 1000)
    x = apply_to_all(subsample, x, 600, 1000)
    emg = x

    for c in FLAGS.remove_channels:
        emg[:,int(c)] = 0
        emg_orig[:,int(c)] = 0

    emg_features = get_emg_features(emg)

    mfccs, audio_discrete = load_audio(os.path.join(base_dir, f'{index}_audio_clean.flac'),
            max_frames=min(emg_features.shape[0], 800 if limit_length else float('inf')))

    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[:mfccs.shape[0],:]
    emg = emg[6:6+6*emg_features.shape[0],:]
    emg_orig = emg_orig[8:8+8*emg_features.shape[0],:]
    assert emg.shape[0] == emg_features.shape[0]*6

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = f'{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid'
    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname, mfccs.shape[0], phoneme_inventory)
    else:
        phonemes = np.zeros(mfccs.shape[0], dtype=np.int64)+phoneme_inventory.index('sil')

    return mfccs, audio_discrete, emg_features, info['text'], (info['book'],info['sentence_index']), phonemes, emg_orig.astype(np.float32)


def read_phonemes(textgrid_fname, mfcc_len, phone_inventory):
    tg = TextGrid(textgrid_fname)
    phone_ids = np.zeros(int(tg['phones'][-1].xmax*100), dtype=np.int64)
    phone_ids[:] = -1
    for interval in tg['phones']:
        phone = interval.text.lower()
        if phone in ['', 'sp', 'spn']:
            phone = 'sil'
        if phone[-1] in string.digits:
            phone = phone[:-1]
        ph_id = phone_inventory.index(phone)
        phone_ids[int(interval.xmin*100):int(interval.xmax*100)] = ph_id
    assert (phone_ids >= 0).all(), 'missing aligned phones'

    phone_ids = phone_ids[1:mfcc_len+1] # mfccs is 2-3 shorter due to edge effects
    return phone_ids

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

class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            directory_info, file_idx = self.dataset.example_indices[idx]
            with open(os.path.join(directory_info.directory, f'{file_idx}_info.json')) as f:
                info = json.load(f)
            if not np.any([l in string.ascii_letters for l in info['text']]):
                continue
            length = sum([emg_len for emg_len, _, _ in info['chunks']])
            if length > self.max_len:
                logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch

class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, limit_length=False, dev=False, test=False, no_testset=False, no_normalizers=False):

        self.text_align_directory = FLAGS.text_align_directory

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

        sample_mfccs, _, sample_emg, _, _, _, _ = load_utterance(self.example_indices[0][0].directory, self.example_indices[0][1])
        self.num_speech_features = sample_mfccs.shape[1]
        self.num_features = sample_emg.shape[1]
        self.limit_length = limit_length
        self.num_sessions = len(directories)

    def silent_subset(self):
        silent_indices = []
        for i, (d, _) in enumerate(self.example_indices):
            if d.silent:
                silent_indices.append(i)
        return torch.utils.data.Subset(self, silent_indices)

    def __len__(self):
        return len(self.example_indices)

    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        mfccs, audio, emg, text, book_location, phonemes, raw_emg = load_utterance(directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory)
        raw_emg = raw_emg / 10

        if not self.no_normalizers:
            mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg)
            emg = 8*np.tanh(emg/8.)

        session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)

        result = {'audio_features':mfccs, 'quantized_audio':audio, 'emg':emg, 'text':text, 'file_label':idx, 'session_ids':session_ids, 'book_location':book_location, 'silent':directory_info.silent, 'raw_emg':raw_emg}

        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, _, voiced_emg, _, _, phonemes, _ = load_utterance(voiced_directory.directory, voiced_idx, False, text_align_directory=self.text_align_directory)

            if not self.no_normalizers:
                voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
                voiced_emg = self.emg_norm.normalize(voiced_emg)
                voiced_emg = 8*np.tanh(voiced_emg/8.)

            result['parallel_voiced_audio_features'] = voiced_mfccs
            result['parallel_voiced_emg'] = voiced_emg

        result['phonemes'] = phonemes # either from this example if vocalized or aligned example if silent

        return result

    @staticmethod
    def collate_fixed_length(batch):
        batch_size = len(batch)
        audio_features = []
        audio_feature_lengths = []
        parallel_emg = []
        for ex in batch:
            if ex['silent']:
                audio_features.append(ex['parallel_voiced_audio_features'])
                audio_feature_lengths.append(ex['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(ex['parallel_voiced_emg'])
            else:
                audio_features.append(ex['audio_features'])
                audio_feature_lengths.append(ex['audio_features'].shape[0])
                parallel_emg.append(np.zeros(1))
        audio_features = [torch.from_numpy(af) for af in audio_features]
        parallel_emg = [torch.from_numpy(pe) for pe in parallel_emg]
        phonemes = [torch.from_numpy(ex['phonemes']) for ex in batch]
        emg = [torch.from_numpy(ex['emg']) for ex in batch]
        raw_emg = [torch.from_numpy(ex['raw_emg']) for ex in batch]
        session_ids = [torch.from_numpy(ex['session_ids']) for ex in batch]
        lengths = [ex['emg'].shape[0] for ex in batch]
        silent = [ex['silent'] for ex in batch]

        seq_len = 200
        result = {'audio_features':combine_fixed_length(audio_features, seq_len),
                  'audio_feature_lengths':audio_feature_lengths,
                  'emg':combine_fixed_length(emg, seq_len),
                  'raw_emg':combine_fixed_length(raw_emg, seq_len*8),
                  'parallel_voiced_emg':parallel_emg,
                  'phonemes':phonemes,
                  'session_ids':combine_fixed_length(session_ids, seq_len),
                  'lengths':lengths,
                  'silent':silent}
        return result

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
    for i in range(1000):
        d[i]


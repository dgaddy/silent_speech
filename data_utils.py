import numpy as np
import librosa
import soundfile as sf

import torch
import matplotlib.pyplot as plt

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('mel_spectrogram', False, 'use mel spectrogram features instead of mfccs for audio')
flags.DEFINE_string('normalizers_file', 'normalizers.pkl', 'file with pickled feature normalizers')

phoneme_inventory = ['aa','ae','ah','ao','aw','ax','axr','ay','b','ch','d','dh','dx','eh','el','em','en','er','ey','f','g','hh','hv','ih','iy','jh','k','l','m','n','nx','ng','ow','oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil']

def normalize_volume(audio):
    rms = librosa.feature.rms(audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms/max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0: # this shouldn't happen too often with the target_rms of 0.2
        audio = audio / max_val
    return audio

def load_audio(filename, start=None, end=None, max_frames=None, renormalize_volume=False):
    audio, r = sf.read(filename)
    assert r == 16000

    if len(audio.shape) > 1:
        audio = audio[:,0] # select first channel of stero audio
    if start is not None or end is not None:
        audio = audio[start:end]

    if renormalize_volume:
        audio = normalize_volume(audio)
    if FLAGS.mel_spectrogram:
        mfccs = librosa.feature.melspectrogram(audio, sr=16000, n_mels=128, center=False, n_fft=512, win_length=432, hop_length=160).T
        mfccs = np.log(mfccs+1e-5)
    else:
        mfccs = librosa.feature.mfcc(audio, sr=16000, n_mfcc=26, n_fft=512, win_length=432, hop_length=160, center=False).T
    audio_discrete = librosa.core.mu_compress(audio, mu=255, quantize=True)+128
    if max_frames is not None and mfccs.shape[0] > max_frames:
        mfccs = mfccs[:max_frames,:]
    audio_length = 160*mfccs.shape[0]+(432-160)
    audio_discrete = audio_discrete[:audio_length] # cut off audio to match framed length
    return mfccs.astype(np.float32), audio_discrete

def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w

def get_emg_features(emg_data, debug=False):
    xs = emg_data - emg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:,i]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(w, frame_length=16, hop_length=6).mean(axis=0)
        p_w = librosa.feature.rms(w, frame_length=16, hop_length=6, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(r, frame_length=16, hop_length=6, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=16, hop_length=6, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=16, hop_length=6).mean(axis=0)

        s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))
        # s has feature dimension first and time second

        if debug:
            plt.subplot(7,1,1)
            plt.plot(x)
            plt.subplot(7,1,2)
            plt.plot(w_h)
            plt.subplot(7,1,3)
            plt.plot(p_w)
            plt.subplot(7,1,4)
            plt.plot(p_r)
            plt.subplot(7,1,5)
            plt.plot(z_p)
            plt.subplot(7,1,6)
            plt.plot(r_h)

            plt.subplot(7,1,7)
            plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')

            plt.show()

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)

class FeatureNormalizer(object):
    def __init__(self, feature_samples, share_scale=False):
        """ features_samples should be list of 2d matrices with dimension (time, feature) """
        feature_samples = np.concatenate(feature_samples, axis=0)
        self.feature_means = feature_samples.mean(axis=0, keepdims=True)
        if share_scale:
            self.feature_stddevs = feature_samples.std()
        else:
            self.feature_stddevs = feature_samples.std(axis=0, keepdims=True)

    def normalize(self, sample):
        sample -= self.feature_means
        sample /= self.feature_stddevs
        return sample

    def inverse(self, sample):
        sample = sample * self.feature_stddevs
        sample = sample + self.feature_means
        return sample

def combine_fixed_length(tensor_list, length):
    total_length = sum(t.size(0) for t in tensor_list)
    if total_length % length != 0:
        pad_length = length - (total_length % length)
        tensor_list = list(tensor_list) # copy
        tensor_list.append(torch.zeros(pad_length,*tensor_list[0].size()[1:], dtype=tensor_list[0].dtype))
        total_length += pad_length
    tensor = torch.cat(tensor_list, 0)
    n = total_length // length
    return tensor.view(n, length, *tensor.size()[1:])

def decollate_tensor(tensor, lengths):
    b, s, d = tensor.size()
    tensor = tensor.view(b*s, d)
    results = []
    idx = 0
    for length in lengths:
        assert idx + length <= b * s
        results.append(tensor[idx:idx+length])
        idx += length
    return results

def splice_audio(chunks, overlap):
    chunks = [c.copy() for c in chunks] # copy so we can modify in place

    assert np.all([c.shape[0]>=overlap for c in chunks])

    result_len = sum(c.shape[0] for c in chunks) - overlap*(len(chunks)-1)
    result = np.zeros(result_len, dtype=chunks[0].dtype)

    ramp_up = np.linspace(0,1,overlap)
    ramp_down = np.linspace(1,0,overlap)

    i = 0
    for chunk in chunks:
        l = chunk.shape[0]

        # note: this will also fade the beginning and end of the result
        chunk[:overlap] *= ramp_up
        chunk[-overlap:] *= ramp_down

        result[i:i+l] += chunk
        i += l-overlap

    return result

def print_confusion(confusion_mat, n=10):
    # axes are (pred, target)
    target_counts = confusion_mat.sum(0) + 1e-4
    aslist = []
    for p1 in range(len(phoneme_inventory)):
        for p2 in range(p1):
            if p1 != p2:
                aslist.append(((confusion_mat[p1,p2]+confusion_mat[p2,p1])/(target_counts[p1]+target_counts[p2]), p1, p2))
    aslist.sort()
    aslist = aslist[-n:]
    max_val = aslist[-1][0]
    min_val = aslist[0][0]
    val_range = max_val - min_val
    print('Common confusions (confusion, accuracy)')
    for v, p1, p2 in aslist:
        p1s = phoneme_inventory[p1]
        p2s = phoneme_inventory[p2]
        print(f'{p1s} {p2s} {v*100:.1f} {(confusion_mat[p1,p1]+confusion_mat[p2,p2])/(target_counts[p1]+target_counts[p2])*100:.1f}')

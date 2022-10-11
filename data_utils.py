import string

import numpy as np
import librosa
import soundfile as sf
from textgrids import TextGrid
import jiwer
from unidecode import unidecode

import torch
import matplotlib.pyplot as plt

from absl import flags
FLAGS = flags.FLAGS
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

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa.filters.mel(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def load_audio(filename, start=None, end=None, max_frames=None, renormalize_volume=False):
    audio, r = sf.read(filename)

    if len(audio.shape) > 1:
        audio = audio[:,0] # select first channel of stero audio
    if start is not None or end is not None:
        audio = audio[start:end]

    if renormalize_volume:
        audio = normalize_volume(audio)
    if r == 16000:
        audio = librosa.resample(audio, 16000, 22050)
    else:
        assert r == 22050
    audio = np.clip(audio, -1, 1) # because resampling sometimes pushes things out of range
    pytorch_mspec = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000, center=False)
    mspec = pytorch_mspec.squeeze(0).T.numpy()
    if max_frames is not None and mspec.shape[0] > max_frames:
        mspec = mspec[:max_frames,:]
    return mspec

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
        tensor_list.append(torch.zeros(pad_length,*tensor_list[0].size()[1:], dtype=tensor_list[0].dtype, device=tensor_list[0].device))
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

def read_phonemes(textgrid_fname, max_len=None):
    tg = TextGrid(textgrid_fname)
    phone_ids = np.zeros(int(tg['phones'][-1].xmax*86.133)+1, dtype=np.int64)
    phone_ids[:] = -1
    phone_ids[-1] = phoneme_inventory.index('sil') # make sure list is long enough to cover full length of original sequence
    for interval in tg['phones']:
        phone = interval.text.lower()
        if phone in ['', 'sp', 'spn']:
            phone = 'sil'
        if phone[-1] in string.digits:
            phone = phone[:-1]
        ph_id = phoneme_inventory.index(phone)
        phone_ids[int(interval.xmin*86.133):int(interval.xmax*86.133)] = ph_id
    assert (phone_ids >= 0).all(), 'missing aligned phones'

    if max_len is not None:
        phone_ids = phone_ids[:max_len]
        assert phone_ids.shape[0] == max_len
    return phone_ids

class TextTransform(object):
    def __init__(self):
        self.transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
        self.chars = string.ascii_lowercase+string.digits+' '

    def clean_text(self, text):
        text = unidecode(text)
        text = self.transformation(text)
        return text

    def text_to_int(self, text):
        text = self.clean_text(text)
        return [self.chars.index(c) for c in text]

    def int_to_text(self, ints):
        return ''.join(self.chars[i] for i in ints)

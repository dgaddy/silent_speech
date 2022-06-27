import os
import json
import sys
import numpy as np

import torch

sys.path.append('./hifi_gan')
from env import AttrDict
from models import Generator

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('hifigan_checkpoint', None, 'filename of hifi-gan generator checkpoint')

class Vocoder(object):
    def __init__(self, device='cuda'):
        assert FLAGS.hifigan_checkpoint is not None
        checkpoint_file = FLAGS.hifigan_checkpoint
        config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            hparams = AttrDict(json.load(f))
        self.generator = Generator(hparams).to(device)
        self.generator.load_state_dict(torch.load(checkpoint_file)['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

    def __call__(self, mel_spectrogram):
        '''
            mel_spectrogram should be a tensor of shape (seq_len, 80)
            returns 1d tensor of audio
        '''
        with torch.no_grad():
            mel_spectrogram = mel_spectrogram.T[np.newaxis,:,:]
            audio = self.generator(mel_spectrogram)
        return audio.squeeze()

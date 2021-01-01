import sys
import os
import numpy as np
import soundfile as sf
import librosa
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from nv_wavenet.pytorch.wavenet import WaveNet
from nv_wavenet.pytorch import nv_wavenet

from data_utils import splice_audio
from read_emg import EMGDataset
from read_librispeech import SpeechDataset

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('output_directory', 'output', 'where to save models and outputs')
flags.DEFINE_boolean('librispeech', False, 'train with librispeech data')
flags.DEFINE_string('pretrained_wavenet_model', None, 'filename of model to start training with')
flags.DEFINE_float('clip_norm', 0.1, 'gradient clipping max norm')
flags.DEFINE_boolean('wavenet_no_lstm', False, "don't use a LSTM before the wavenet")

class WavenetModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if not FLAGS.wavenet_no_lstm:
            self.lstm = nn.LSTM(input_dim, 512, bidirectional=True, batch_first=True)
            self.projection_layer = nn.Linear(512*2, 128)
        else:
            self.projection_layer = nn.Linear(input_dim, 128)
        self.wavenet = WaveNet(n_in_channels=256, n_layers=16, max_dilation=128, n_residual_channels=64, n_skip_channels=256, n_out_channels=256, n_cond_channels=128, upsamp_window=432, upsamp_stride=160)

    def pre_wavenet_processing(self, x):
        if not FLAGS.wavenet_no_lstm:
            x, _ = self.lstm(x)
            x = F.dropout(x, 0.5, training=self.training)
        x = self.projection_layer(x)
        return x.permute(0,2,1)

    def forward(self, x, audio):
        x = self.pre_wavenet_processing(x)
        return self.wavenet((x, audio))

def test(wavenet_model, testset, device):
    wavenet_model.eval()

    errors = []
    dataloader = torchdata.DataLoader(testset, batch_size=1, shuffle=True, pin_memory=(device=='cuda'))
    with torch.no_grad():
        for batch in dataloader:
            mfcc = batch['audio_features'].to(device)
            audio = batch['quantized_audio'].to(device)

            audio_out = wavenet_model(mfcc, audio)
            loss = F.cross_entropy(audio_out, audio)
            errors.append(loss.item())

    wavenet_model.train()
    return np.mean(errors)

def save_output(wavenet_model, input_data, filename, device):
    wavenet_model.eval()

    assert len(input_data.shape) == 2
    X = torch.tensor(input_data, dtype=torch.float32).to(device).unsqueeze(0)

    wavenet = wavenet_model.wavenet
    inference_wavenet = nv_wavenet.NVWaveNet(**wavenet.export_weights())
    cond_input = wavenet_model.pre_wavenet_processing(X)

    chunk_len = 400
    overlap = 1
    audio_chunks = []
    for i in range(0, cond_input.size(2), chunk_len-overlap):
        if cond_input.size(2)-i < overlap:
            break # don't make segment at end that doesn't go past overlapped part
        cond_chunk = cond_input[:,:,i:i+chunk_len]
        wavenet_cond_input = wavenet.get_cond_input(cond_chunk)
        audio_data = inference_wavenet.infer(wavenet_cond_input, nv_wavenet.Impl.SINGLE_BLOCK)
        audio_chunk = librosa.core.mu_expand(audio_data.squeeze(0).cpu().numpy()-128, 255, True)
        audio_chunks.append(audio_chunk)

    audio_out = splice_audio(audio_chunks, overlap*160)

    sf.write(filename, audio_out, 16000)

    wavenet_model.train()

def train():
    if FLAGS.librispeech:
        dataset = SpeechDataset('LibriSpeech/train-clean-100-sliced', 'M', 'LibriSpeech/SPEAKERS.TXT')
        testset = torch.utils.data.Subset(dataset, list(range(10)))
        trainset = torch.utils.data.Subset(dataset, list(range(10,len(dataset))))
        num_features = dataset.num_speech_features
        batch_size = 4
        logging.info('output example: %s', dataset.filenames[0])
    else:
        trainset = EMGDataset(dev=False, test=False, limit_length=True)
        testset = EMGDataset(dev=True, limit_length=True)
        num_features = testset.num_speech_features
        batch_size = 1
        logging.info('output example: %s', testset.example_indices[0])

    if not os.path.exists(FLAGS.output_directory):
        os.makedirs(FLAGS.output_directory)

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'

    wavenet_model = WavenetModel(num_features).to(device)
    if FLAGS.pretrained_wavenet_model is not None:
        wavenet_model.load_state_dict(torch.load(FLAGS.pretrained_wavenet_model))

    optim = torch.optim.Adam(wavenet_model.parameters(), weight_decay=1e-7)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=2)
    dataloader = torchdata.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=(device=='cuda'))

    best_dev_err = float('inf')
    for epoch_idx in range(50):
        losses = []
        for batch in dataloader:
            mfcc = batch['audio_features'].to(device)
            audio = batch['quantized_audio'].to(device)

            optim.zero_grad()

            audio_out = wavenet_model(mfcc, audio)
            loss = F.cross_entropy(audio_out, audio)

            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(wavenet_model.parameters(), FLAGS.clip_norm)
            optim.step()
        train_err = np.mean(losses)
        dev_err = test(wavenet_model, testset, device)
        lr_sched.step(dev_err)
        logging.info(f'finished epoch {epoch_idx+1} with error {dev_err:.2f}')
        logging.info(f' train error {train_err:.2f}')
        if dev_err < best_dev_err:
            logging.info('saving model')
            torch.save(wavenet_model.state_dict(), os.path.join(FLAGS.output_directory, 'wavenet_model.pt'))
            best_dev_err = dev_err

    wavenet_model.load_state_dict(torch.load(os.path.join(FLAGS.output_directory,'wavenet_model.pt'))) # re-load best parameters
    for i, datapoint in enumerate(testset):
        save_output(wavenet_model, datapoint['audio_features'], os.path.join(FLAGS.output_directory, f'wavenet_output_{i}.wav'), device)

if __name__ == '__main__':
    FLAGS(sys.argv)

    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")
    logging.info(sys.argv)

    train()

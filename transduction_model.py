import os
import sys
import numpy as np
import logging
import subprocess
import random

import soundfile as sf

import torch
from torch import nn
import torch.nn.functional as F

from read_emg import EMGDataset, SizeAwareSampler
from architecture import Model
from align import align_from_distances
from asr_evaluation import evaluate
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
from vocoder import Vocoder

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'training batch size')
flags.DEFINE_integer('epochs', 80, 'number of training epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_integer('learning_rate_warmup', 500, 'steps of linear warmup')
flags.DEFINE_string('start_training_from', None, 'start training from this model')
flags.DEFINE_float('data_size_fraction', 1.0, 'fraction of training data to use')
flags.DEFINE_float('phoneme_loss_weight', 0.5, 'weight of auxiliary phoneme prediction loss')
flags.DEFINE_float('l2', 1e-7, 'weight decay')
flags.DEFINE_string('output_directory', 'output', 'output directory')

def test(model, testset, device):
    model.eval()

    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, collate_fn=testset.collate_raw)
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(phoneme_inventory),len(phoneme_inventory)))
    seq_len = 200
    with torch.no_grad():
        for batch in dataloader:
            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len*8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            pred, phoneme_pred = model(X, X_raw, sess)

            loss, phon_acc = dtw_loss(pred, phoneme_pred, batch, True, phoneme_confusion)
            losses.append(loss.item())

            accuracies.append(phon_acc)

    model.train()
    return np.mean(losses), np.mean(accuracies), phoneme_confusion #TODO size-weight average

def save_output(model, datapoint, filename, device, audio_normalizer, vocoder):
    model.eval()
    with torch.no_grad():
        sess = torch.tensor(datapoint['session_ids'], device=device).unsqueeze(0)
        X = torch.tensor(datapoint['emg'], dtype=torch.float32, device=device).unsqueeze(0)
        X_raw = torch.tensor(datapoint['raw_emg'], dtype=torch.float32, device=device).unsqueeze(0)

        pred, _ = model(X, X_raw, sess)
        y = pred.squeeze(0)

        y = audio_normalizer.inverse(y.cpu()).to(device)

        audio = vocoder(y).cpu().numpy()

    sf.write(filename, audio, 22050)

    model.train()

def get_aligned_prediction(model, datapoint, device, audio_normalizer):
    model.eval()
    with torch.no_grad():
        silent = datapoint['silent']
        sess = datapoint['session_ids'].to(device).unsqueeze(0)
        X = datapoint['emg'].to(device).unsqueeze(0)
        X_raw = datapoint['raw_emg'].to(device).unsqueeze(0)
        y = datapoint['parallel_voiced_audio_features' if silent else 'audio_features'].to(device).unsqueeze(0)

        pred, _ = model(X, X_raw, sess) # (1, seq, dim)

        if silent:
            costs = torch.cdist(pred, y).squeeze(0)
            alignment = align_from_distances(costs.T.detach().cpu().numpy())
            pred_aligned = pred.squeeze(0)[alignment]
        else:
            pred_aligned = pred.squeeze(0)

        pred_aligned = audio_normalizer.inverse(pred_aligned.cpu())

    model.train()
    return pred_aligned

def dtw_loss(predictions, phoneme_predictions, example, phoneme_eval=False, phoneme_confusion=None):
    device = predictions.device

    predictions = decollate_tensor(predictions, example['lengths'])
    phoneme_predictions = decollate_tensor(phoneme_predictions, example['lengths'])

    audio_features = [t.to(device, non_blocking=True) for t in example['audio_features']]

    phoneme_targets = example['phonemes']

    losses = []
    correct_phones = 0
    total_length = 0
    for pred, y, pred_phone, y_phone, silent in zip(predictions, audio_features, phoneme_predictions, phoneme_targets, example['silent']):
        assert len(pred.size()) == 2 and len(y.size()) == 2
        y_phone = y_phone.to(device)

        if silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
            costs = dists.squeeze(0)

            # pred_phone (seq1_len, 48), y_phone (seq2_len)
            # phone_probs (seq1_len, seq2_len)
            pred_phone = F.log_softmax(pred_phone, -1)
            phone_lprobs = pred_phone[:,y_phone]

            costs = costs + FLAGS.phoneme_loss_weight * -phone_lprobs

            alignment = align_from_distances(costs.T.cpu().detach().numpy())

            loss = costs[alignment,range(len(alignment))].sum()

            if phoneme_eval:
                alignment = align_from_distances(costs.T.cpu().detach().numpy())

                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == y_phone).sum().item()

                for p, t in zip(pred_phone[alignment].tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1
        else:
            assert y.size(0) == pred.size(0)

            dists = F.pairwise_distance(y, pred)

            assert len(pred_phone.size()) == 2 and len(y_phone.size()) == 1
            phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction='sum')
            loss = dists.sum() + FLAGS.phoneme_loss_weight * phoneme_loss

            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone == y_phone).sum().item()

                for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1

        losses.append(loss)
        total_length += y.size(0)

    return sum(losses)/total_length, correct_phones/total_length

def train_model(trainset, devset, device, save_sound_outputs=True):
    n_epochs = FLAGS.epochs

    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = trainset.subset(FLAGS.data_size_fraction)
    dataloader = torch.utils.data.DataLoader(training_subset, pin_memory=(device=='cuda'), collate_fn=devset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(training_subset, 256000))

    n_phones = len(phoneme_inventory)
    model = Model(devset.num_features, devset.num_speech_features, n_phones).to(device)

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model.load_state_dict(state_dict, strict=False)

    if save_sound_outputs:
        vocoder = Vocoder()

    optim = torch.optim.AdamW(model.parameters(), weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=FLAGS.learning_rate_patience)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)

    seq_len = 200

    batch_idx = 0
    for epoch_idx in range(n_epochs):
        losses = []
        for batch in dataloader:
            optim.zero_grad()
            schedule_lr(batch_idx)

            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len*8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            pred, phoneme_pred = model(X, X_raw, sess)

            loss, _ = dtw_loss(pred, phoneme_pred, batch)
            losses.append(loss.item())

            loss.backward()
            optim.step()

            batch_idx += 1
        train_loss = np.mean(losses)
        val, phoneme_acc, _ = test(model, devset, device)
        lr_sched.step(val)
        logging.info(f'finished epoch {epoch_idx+1} - validation loss: {val:.4f} training loss: {train_loss:.4f} phoneme accuracy: {phoneme_acc*100:.2f}')
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory,'model.pt'))
        if save_sound_outputs:
            save_output(model, devset[0], os.path.join(FLAGS.output_directory, f'epoch_{epoch_idx}_output.wav'), device, devset.mfcc_norm, vocoder)

    if save_sound_outputs:
        for i, datapoint in enumerate(devset):
            save_output(model, datapoint, os.path.join(FLAGS.output_directory, f'example_output_{i}.wav'), device, devset.mfcc_norm, vocoder)

        evaluate(devset, FLAGS.output_directory)

    return model

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    logging.info(subprocess.run(['git','rev-parse','HEAD'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(subprocess.run(['git','diff'], stdout=subprocess.PIPE, universal_newlines=True).stdout)

    logging.info(sys.argv)

    trainset = EMGDataset(dev=False,test=False)
    devset = EMGDataset(dev=True)
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d',len(trainset),len(devset))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = train_model(trainset, devset, device, save_sound_outputs=(FLAGS.hifigan_checkpoint is not None))

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()

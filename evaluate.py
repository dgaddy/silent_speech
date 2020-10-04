import sys
import os

import torch
from torch import nn

from transduction_model import test, save_output, Model
from read_emg import EMGDataset
from asr import evaluate

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('models', [], 'identifiers of models to evaluate')

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, sess):
        ys = []
        for model in self.models:
            ys.append(model(x, sess))
        return torch.mean(torch.stack(ys,0),dim=0)

def main():
    testset = EMGDataset(test=True)

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'

    models = []
    for num in FLAGS.models:
        fname  = f'./models/{num}/model.pt'
        state_dict = torch.load(fname)
        n_sess = 1 if FLAGS.no_session_embed else state_dict["session_emb.weight"].size(0)
        model = Model(testset.num_features, testset.num_speech_features, n_sess).to(device)
        model.load_state_dict(state_dict)
        models.append(model)
    ensemble = EnsembleModel(models)

    error = test(ensemble, testset, device)
    print(error)

    os.makedirs(FLAGS.output_directory, exist_ok=True)
    for i, datapoint in enumerate(testset):
        save_output(ensemble, datapoint, os.path.join(FLAGS.output_directory, f'example_output_{i}.wav'), device)

    evaluate(testset, FLAGS.output_directory)

if __name__ == "__main__":
    FLAGS(sys.argv)
    main()

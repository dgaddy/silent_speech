# Digital Voicing of Silent Speech

## Data

The necessary data can be downloaded from https://doi.org/10.5281/zenodo.4064408.

## Environment Setup

This code requires Python 3.6 or later.
We strongly recommend running in a new Anaconda environment.

First we will do some conda installs.  Your environment must use CUDA 10.1 exactly, since DeepSpeech was compiled with this version.
```
conda install cudatoolkit=10.1
conda install pytorch -c pytorch
conda install libsndfile=1.0.28 -c conda-forge
```

Pull nv-wavenet into the `nv_wavenet` folder and follow build intructions provided in the repository.
```
git clone https://github.com/NVIDIA/nv-wavenet.git nv_wavenet
# follow build instructions in nv_wavenet/pytorch
```
If you have an older GPU, you may need to use https://github.com/dgaddy/nv-wavenet instead, which removes code for 16-bit floating point that only works on newer GPUs.

The rest of the required packages can be installed with pip.
```
pip install absl-py librosa soundfile matplotlib scipy scikit-learn numba jiwer unidecode deepspeech==0.8.2
```

Download pre-trained DeepSpeech model files.  It is important that you use DeepSpeech version 0.7.0 model files to maintain consistency of evaluation.  Note that the DeepSpeech pip package we recommend is version 0.8.2 (which uses a more up-to-date CUDA), but this is compatible with version 0.7.x model files.
```
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.scorer
```

## Running

To train a WaveNet model, use
```
python wavenet_model.py --output_directory "./models/wavenet_model/" --voiced_data_directories "./emg_data/voiced_parallel_data,./emg_data/nonparallel_data"
```

To train an EMG to speech feature transduction model, use
```
python transduction_model.py --pretrained_wavenet_model "./models/wavenet_model/wavenet_model.pt" --output_directory "./models/transduction_model/" --voiced_data_directories "./emg_data/voiced_parallel_data,./emg_data/nonparallel_data" --silent_data_directories "./emg_data/silent_parallel_data"
```
At the end of training, an ASR evaluation will be run on the validation set.

Finally, to evaluate a model on the test set, use
```
python evaluate.py --models transduction_model --pretrained_wavenet_model ./models/wavenet_model/wavenet_model.pt --silent_data_directories ./emg_data/silent_parallel_data --voiced_data_directories ./emg_data/voiced_parallel_data --output_directory evaluation_output
```

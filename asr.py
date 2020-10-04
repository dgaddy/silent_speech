import os

import deepspeech
import jiwer
import soundfile as sf
import numpy as np
from unidecode import unidecode

def evaluate(testset, audio_directory):
    model = deepspeech.Model('deepspeech-0.7.0-models.pbmm')
    model.enableExternalScorer('deepspeech-0.7.0-models.scorer')
    predictions = []
    targets = []
    for i, datapoint in enumerate(testset):
        audio, rate = sf.read(os.path.join(audio_directory,f'example_output_{i}.wav'))
        assert rate == model.sampleRate(), 'wrong sample rate'
        audio_int16 = (audio*(2**15)).astype(np.int16)
        text = model.stt(audio_int16)
        predictions.append(text)
        target_text = unidecode(datapoint['text'])
        targets.append(target_text)
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    print('targets:', targets)
    print('predictions:', predictions)
    print('wer:', jiwer.wer(targets, predictions))

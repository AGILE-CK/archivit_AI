import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
import tempfile
from pydub import AudioSegment
import os


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits



# load model from hub
device = device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)



def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

def interpret_emotion(prediction):
    arousal, dominance, valence = prediction[0]

    # Calm situation: Low arousal, Moderate-High dominance, Moderate-High valence
    if arousal < 0.4 and dominance > 0.4 and valence > 0.4:
        return "Calm Situation"

    # Violent or Aggressive situation: High arousal, Low-High dominance, Low valence
    if arousal > 0.6 and valence < 0.4:
        return "Violent or Aggressive Situation"

    # Default or undefined situation
    return "Default Situation"

def convert_to_wav(file_path):
    """Convert an audio file to WAV format if necessary."""
        # Check if the file is already a WAV file
    file, extension = os.path.splitext(file_path)
    if extension.lower() == '.wav':
        # File is already a WAV, no need to convert
        return file_path
    sound = AudioSegment.from_file(file_path)
    wav_file_path = tempfile.mktemp(suffix=".wav")
    sound.export(wav_file_path, format="wav")
    return wav_file_path


def predict_emotion(audio_data):
    wav_data = convert_to_wav(audio_data)
    signal, _ = librosa.load(wav_data, sr=16000, mono=True, dtype=np.float32)
    emotion_prediction = process_func(signal, 16000)  # Ensure sampling rate is 16000
    situation = interpret_emotion(emotion_prediction)
    
    # Check if a temporary WAV file was created and delete it
    if wav_data != audio_data:
        os.remove(wav_data)  # Delete the temporary WAV file
    
    return situation
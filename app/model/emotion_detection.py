import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


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

# dummy signal
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)


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

import pyaudio
import wave
import librosa
import numpy as np

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # Adjust if necessary
RECORD_SECONDS = 1  # Duration of each recording snippet
TOTAL_DURATION = 200  # Total duration to run the loop

audio = pyaudio.PyAudio()

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

# Function to process and predict emotion from audio snippet
def predict_emotion(audio_data):
    signal, sr = librosa.load(audio_data, sr=RATE, mono=True, dtype=np.float32)
    emotion_prediction = process_func(signal, sr)
    situation = interpret_emotion(emotion_prediction)
    print(situation)
    return situation

# Main loop
def is_calm():
    for _ in range(0, int(TOTAL_DURATION / RECORD_SECONDS)):
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Recording...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Save the recorded data as a WAV file
        file_name = "temp_recording_emotion.wav"
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Predict emotion and stop if calm situation detected
        situation = predict_emotion(file_name)
        if situation == "Calm Situation":
            print("Calm situation detected, stopping.")
            break
    return situation

    audio.terminate()
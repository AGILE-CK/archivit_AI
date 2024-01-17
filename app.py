import os
import pyaudio
import queue
import threading
from google.cloud import speech
import torch
from openprompt.data_utils import InputExample
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer, ManualTemplate
from openprompt.pipeline_base import PromptForClassification
from transformers import RobertaTokenizer

# Set the path to your Google credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'speech-to-text-key.json'

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms




def predict_new_sentence(sentence, model, template, tokenizer, device, WrapperClass):
    # Create an InputExample object
    input_example = InputExample(text_a=sentence, label=0)  # Label is dummy
    
    # Create a DataLoader for the single example
    single_example_dataloader = PromptDataLoader(
        dataset=[input_example], 
        template=template, 
        tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, 
        batch_size=1, 
        shuffle=False,
        max_seq_length=128
    )

    # Move model to the appropriate device
    model.to(device)
    model.eval()

    # Forward pass: compute logits
    with torch.no_grad():
        for batch in single_example_dataloader:
            # Move batch to the appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)

    # Get the class with the highest probability
    predicted_class = torch.argmax(probs, dim=1).item()

    return predicted_class

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self.stream = None
        self.audio_interface = pyaudio.PyAudio()

    def __enter__(self):
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )
        return self

    def __exit__(self, type, value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.audio_interface.terminate()

    def generator(self):
        while True:
            yield self.stream.read(self.chunk)

def listen_print_loop(responses, model, template, tokenizer, device, WrapperClass):
    """Iterates through server responses and prints them."""
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        print(transcript)

        # Check if the transcript is hate speech
        is_hate_speech = predict_new_sentence(transcript, model, template, tokenizer, device, WrapperClass)

        if is_hate_speech == 1:
            print("Hate speech detected!")
        else:
            print("Not hate speech")

def main(tokenizer, model, template, device, WrapperClass):
    # Create a speech client
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='en-US',
        model='default',
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses, model, template, tokenizer, device, WrapperClass)

if __name__ == '__main__':
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('my_tokenizer')

    # Load the PLM, tokenizer, and wrapper class
    plm, _, _, WrapperClass = load_plm("roberta", "roberta-large")

    # Define the template (recreate it as it was during training)
    template = ManualTemplate(
        tokenizer=tokenizer, 
        text='{"placeholder":"text_a"} Sentence for analyze: This sentence is {"mask"}.' 
    )

    # Define the verbalizer (recreate it as it was during training)
    verbalizer = ManualVerbalizer(
        tokenizer=tokenizer, 
        num_classes=2, 
        label_words=['positive','negative']
    )

    # Create the model instance
    model = PromptForClassification(plm, template, verbalizer)

    # Load the saved model state
    model.load_state_dict(torch.load('best_model_by_template.pt'))

    # Define the device (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(tokenizer, model, template, device, WrapperClass)

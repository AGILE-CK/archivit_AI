import sounddevice as sd
import numpy as np
import wave
import torch
import os
import re
import emoji
from soynlp.normalizer import repeat_normalize
import queue
import threading
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Initialize models and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

model_speech = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

tokenizer_violent = AutoTokenizer.from_pretrained("./model/tokenizer_violent_detection")
model_violent = AutoModelForSequenceClassification.from_pretrained("./model/model_violent_detection").to(device)    

processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_speech,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)  

# Function to preprocess text
def text_preprocess(text):
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    text = pattern.sub(' ', text)
    text = emoji.replace_emoji(text, replace='') 
    text = url_pattern.sub('', text)
    text = text.strip()
    text = repeat_normalize(text, num_repeats=2)
    return text

def predict_and_classify_violence(sentence):
    labels = ["common", "sex", "race", "religion", "age", "political", "hate_speech", "origin", "appearance", "sexual_minority", "other_hate"]
    violent_categories = ["hate_speech", "sex", "race", "other_hate", "appearance", "sexual_minority", "religion", "age", "political", "origin"]
    non_violent_categories = ["common"]
    preprocessed_text = text_preprocess(sentence)
    inputs = tokenizer_violent(preprocessed_text, return_tensors="pt", truncation=True, max_length=512)

    # Move the input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model_violent.eval() 
    with torch.no_grad():
        outputs = model_violent(**inputs)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_class = labels[predicted_class_idx]

    # Classify as violent or non-violent
    if predicted_class in violent_categories:
        return "Violent Situation Detected"
    elif predicted_class in non_violent_categories:
        return "Non-Violent Situation"
    else:
        return "Unclassified"

def record_audio(duration=3, samplerate=16000):
    """Record audio for a given duration and samplerate."""
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return audio

def save_to_wav(audio, filename, samplerate=16000):
    """Save the audio to a WAV file."""
    audio = (audio * np.iinfo(np.int16).max).astype(np.int16)  # Convert to 16-bit PCM format
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

def convert_to_wav(file_path, original_suffix):
    """Convert an audio file to WAV format if necessary."""
    if original_suffix in [".wav", ".wave"]:
        return file_path  # No conversion needed for WAV files

    sound = AudioSegment.from_file(file_path)
    wav_file_path = file_path.rsplit('.', 1)[0] + ".wav"
    sound.export(wav_file_path, format="wav")
    return wav_file_path

def transcribe_audio(filename):
    """Transcribe the audio file using the speech recognition pipeline."""
    try:
        result = pipe(filename, generate_kwargs={"language": "korean"})
        return result['text']
    except Exception as e:
        print("Error in processing audio:", e)
        return None

transcription_queue = queue.Queue()
stop_processing = threading.Event()  # Added to signal stopping of processing

def transcribe_worker():
    global violence_detected
    violence_detected = ""
    while not stop_processing.is_set():
        filename = transcription_queue.get()
        if filename is None:
            transcription_queue.task_done()
            break
        transcription = transcribe_audio(filename)
        if transcription:
            print("Transcription:", transcription)
            violence_status = predict_and_classify_violence(transcription)
            print("Violence Analysis:", violence_status)
            if "Violent Situation Detected" in violence_status:
                stop_processing.set()  # Signal to stop processing
                violence_detected = violence_status
        os.remove(filename)
        transcription_queue.task_done()

def is_violent():
    threading.Thread(target=transcribe_worker, daemon=True).start()
    record_index = 0
    while not stop_processing.is_set():
        audio = record_audio()
        temp_filename = f"temp_audio_{record_index}.wav"
        save_to_wav(audio, temp_filename)
        transcription_queue.put(temp_filename)
        record_index += 1
    if violence_detected == "Violent Situation Detected":
        # Reset the event
        stop_processing.clear()
        return violence_detected
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Load the diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")  # Replace with your actual Hugging Face auth token

# Ensure the diarization pipeline is sent to GPU if available
device1 = "cuda" if torch.cuda.is_available() else "cpu"
diarization_pipeline.to(torch.device(device1))




device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

model_speech = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

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

def convert_to_wav(audio_file_path):
    """Converts non-WAV audio file to WAV format and marks it for deletion."""
    converted = False
    wav_path = audio_file_path
    if not audio_file_path.lower().endswith('.wav'):
        sound = AudioSegment.from_file(audio_file_path)
        wav_path = audio_file_path.rsplit('.', 1)[0] + '.wav'
        sound.export(wav_path, format="wav")
        converted = True
    return wav_path, converted

MIN_SEGMENT_DURATION = 1.0  # Minimum segment duration in seconds

def segment_and_transcribe(filename):
    file_converted, converted = convert_to_wav(filename)
    diarization_result = diarization_pipeline(file_converted)
    full_audio = AudioSegment.from_wav(file_converted)
    transcriptions = []

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segment_duration = turn.end - turn.start
        if segment_duration < MIN_SEGMENT_DURATION:
            continue  # Skip segments shorter than the threshold
        
        start_ms, end_ms = turn.start * 1000, turn.end * 1000
        audio_segment = full_audio[start_ms:end_ms]
        segment_filename = f"temp_segment_{speaker}_{int(start_ms)}_{int(end_ms)}.wav"
        audio_segment.export(segment_filename, format="wav")
        
        transcription = transcribe_audio(segment_filename)
        transcriptions.append((speaker, transcription))
        os.remove(segment_filename)

    if converted:
        os.remove(file_converted)
    return transcriptions


def transcribe_audio(filename):
    """Transcribe the audio file using the speech recognition pipeline."""
    try:
        result = pipe(filename, generate_kwargs={"language": "korean"})
        return result['text']
    except Exception as e:
        print("Error in processing audio:", e)
        return None


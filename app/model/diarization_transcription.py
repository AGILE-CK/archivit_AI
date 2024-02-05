from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import os

from model.violent_prediction import transcribe_audio

# Load the diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")  # Replace with your actual Hugging Face auth token

# Ensure the diarization pipeline is sent to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
diarization_pipeline.to(torch.device(device))

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

def segment_and_transcribe(filename):
    """Segments, transcribes audio file using diarization, and cleans up."""
    file_converted, converted = convert_to_wav(filename)

    # Apply the diarization pipeline
    diarization_result = diarization_pipeline(file_converted)

    # Load the full audio file
    full_audio = AudioSegment.from_wav(file_converted)
    transcriptions = []

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # Extract the segment from the full audio
        start_ms = turn.start * 1000
        end_ms = turn.end * 1000
        audio_segment = full_audio[start_ms:end_ms]
        
        # Save the segment to a temporary file
        segment_filename = f"temp_segment_{speaker}_{int(start_ms)}_{int(end_ms)}.wav"
        audio_segment.export(segment_filename, format="wav")
        
        # Transcribe the segment
        transcription = transcribe_audio(segment_filename)
        
        # Append the transcription with speaker label
        transcriptions.append((speaker, transcription))
        
        # Delete the temporary segment file
        os.remove(segment_filename)
    
    # Cleanup: remove the original (if uploaded in a non-WAV format and converted) and converted files
    if converted:
        os.remove(file_converted)
    os.remove(filename)  # Ensure the original file is also removed if it's a temporary upload

    return transcriptions

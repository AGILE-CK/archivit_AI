from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from model.violent_prediction import convert_to_wav, transcribe_audio, predict_and_classify_violence
import uvicorn
from model.emotion_detection import predict_emotion
from pydantic import BaseModel
from model.diarization_transcription import segment_and_transcribe
from pydantic import BaseModel
import tempfile
from model.summarization import generate_content


app = FastAPI()

class TextIn(BaseModel):
    text: str

class PromptRequest(BaseModel):
    prompt_parts: list

@app.post("/summarize/")
def generate(prompt_request: PromptRequest):
    try:
        text = generate_content(prompt_request.prompt_parts)
        return {"generated_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/violent-speech-detection/")
async def violent_speech_detection(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Convert to WAV format for consistency
        wav_file_path = convert_to_wav(temp_file_path)

        # Transcribe the audio file
        transcription = transcribe_audio(wav_file_path)

        # Check for violence in the transcription
        violence_status = predict_and_classify_violence(transcription)

        # Cleanup the temporary files
        os.remove(temp_file_path)
        os.remove(wav_file_path)

        return {"violence_status": violence_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/calm-situation-detection/")
# async def calm_situation_detection():
#     calm = is_calm()
#     return {"status": calm}



    
@app.post("/calm-situation-detection/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Save temporary audio file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        temp_file_path = temp_file.name
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()  # Close the file so it can be accessed by other functions
        
        # Predict emotion
        situation = predict_emotion(temp_file_path)
        
        # Cleanup: Delete the temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return JSONResponse(status_code=200, content={"situation": situation})
    except Exception as e:
        # Ensure cleanup even if an error occurs
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})




@app.post("/transcribe/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Save temporary file
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process file: diarization and transcription
        transcriptions = segment_and_transcribe(temp_file_path)
        
        # Cleanup: Ensure the temporary uploaded file is deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return JSONResponse(content={"transcriptions": transcriptions}, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
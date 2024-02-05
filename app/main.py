from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from model.violent_prediction import is_violent
import uvicorn
from model.emotion_detection import is_calm
from pydantic import BaseModel
from model.diarization_transcription import segment_and_transcribe
from model.summarization import Summarizer
from pydantic import BaseModel



app = FastAPI()

class TextIn(BaseModel):
    text: str

summarizer = Summarizer('traintogpb/pko-t5-large-kor-for-colloquial-summarization-finetuned')

@app.post("/summarize/")
async def create_summary(request: TextIn):
    try:
        summary = summarizer.generate_summary(request.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/violent-speech-detection/")
async def violent_speech_detection():
    violence = is_violent()
    return {"status": violence}

@app.post("/calm-situation-detection/")
async def calm_situation_detection():
    calm = is_calm()
    return {"status": calm}

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
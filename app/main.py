from fastapi import FastAPI
from model.violent_prediction import is_violent
import uvicorn
from model.emotion_detection import is_calm

app = FastAPI()

@app.post("/violent-speech-detection/")
async def violent_speech_detection():
    violence = is_violent()
    return {"status": violence}

@app.post("/calm-situation-detection/")
async def calm_situation_detection():
    calm = is_calm()
    return {"status": calm}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
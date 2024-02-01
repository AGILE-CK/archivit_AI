from fastapi import FastAPI
from model.violent_prediction import is_violent
import uvicorn

app = FastAPI()

@app.post("/analyse-violent-speech/")
async def analyze_speech_endpoint():
    violence = is_violent()
    return {"status": violence}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
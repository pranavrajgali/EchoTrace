from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from core.inference import run_inference

app = FastAPI()

# THIS IS CRITICAL: Without this, React will be blocked by the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Your React URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = await run_inference(contents)
    return result       
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from model import llm
from utils import classify_image

app = FastAPI()

@app.post("/generate")
async def generate(image: UploadFile):
    classified_image = await classify_image(image)
    recipe = llm.invoke({"food": classified_image}).model_dump_json()
    return JSONResponse(recipe)

@app.get("/")
def root():
    return "hey!"

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import replicate
import os
import shutil
import time
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://jobodega.webflow.io", "https://www.jobodega.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

API_BASE_URL = "https://replicate-fastapi.onrender.com"

@app.post("/gerar-headshot")
async def gerar_headshot(
    image: UploadFile = File(...),
    gender: str = Form(...),
    profession: str = Form(...),
    age: str = Form(...),
    clothing: str = Form(...),
    background: List[str] = Form(...)
):
    try:
        # Salvar imagem
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_path = f"temp/{img_id}.{file_ext}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        image_urls = []

        for i, bg in enumerate(background[:5]):
            prompt = f"""
            Ultra high-resolution LinkedIn headshot of a {gender},
            working as a {profession}, aged {age}, 
            wearing {clothing}, confident and approachable.
            Background: {bg}. DSLR style, shallow depth of field, studio lighting.
            """

            prediction = replicate.run(
                "black-forest-labs/flux-kontext-pro:ce33da8b07e77f27a10556bffec8dd0d67d7b23f529801eb4a418b6bcaed7bb5",
                input={
                    "prompt": prompt,
                    "input_image": open(input_path, "rb"),
                    "output_format": "jpg"
                }
            )

            output_path = f"temp/{img_id}_{i}_output.jpg"
            with open(output_path, "wb") as f:
                f.write(prediction.read())

            image_urls.append(f"{API_BASE_URL}/temp/{img_id}_{i}_output.jpg")
            time.sleep(1)

        return {"image_urls": image_urls}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

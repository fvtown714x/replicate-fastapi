from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import replicate
import os
import time
import shutil
from dotenv import load_dotenv
from uuid import uuid4
import json

load_dotenv()

app = FastAPI()

API_BASE_URL = "https://replicate-fastapi.onrender.com"

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://jobodega.webflow.io"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pasta pública para imagens temporárias
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

@app.post("/gerar-headshot")
async def gerar_headshot(
    image: UploadFile = File(...),
    clothing: str = Form(...),
    background: str = Form(...),
    profession: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...)
):
    try:
        clothing_list = json.loads(clothing)
        background_list = json.loads(background)

        if len(clothing_list) != len(background_list):
            return JSONResponse(status_code=400, content={"erro": "Você deve selecionar o mesmo número de roupas e fundos."})

        # Salvar imagem temporária
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_filename = f"{img_id}.{file_ext}"
        input_path = f"temp/{input_filename}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        input_url = f"{API_BASE_URL}/temp/{input_filename}"

        urls = []

        for idx, (clothe, bg) in enumerate(zip(clothing_list, background_list)):
            prompt = (
                f"Professional headshot of a {age}-year-old {gender.lower()} {profession}, "
                f"wearing {clothe}, background: {bg}. "
                f"High-quality DSLR style, well-lit, studio background."
            )

            output = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input=input_data
            )
            
            # `output` é uma URL ou lista de URLs
            if isinstance(output, list):
                urls.extend(output)
            else:
                urls.append(output)

            time.sleep(0.3)

        return {"image_urls": urls}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

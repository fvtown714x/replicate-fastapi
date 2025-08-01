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

load_dotenv()

app = FastAPI()

API_BASE_URL = "https://replicate-fastapi.onrender.com"

# CORS para testes locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste para seu domínio Webflow depois
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Criar e expor pasta temp para acesso público
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")@app.post("/gerar-headshot")
async def gerar_headshot(
    image: UploadFile = File(...),
    clothing: str = Form(...),
    background: str = Form(...),
    profession: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...)
):
    try:
        # Salvar imagem temporária
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_path = f"temp/{img_id}.{file_ext}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Construir o prompt base
        prompt_base = (
            f"Professional headshot of a {age}-year-old {gender.lower()} {profession}, "
            f"wearing {clothing}, background: {background}. "
            f"High-quality DSLR style, well-lit, studio background."
        )

        # Gerar 5 variações
        urls = []
        for i in range(5):
            prompt = f"{prompt_base} (variation {i+1})"
            input_data = {
                "prompt": prompt,
                "input_image": open(input_path, "rb"),
                "output_format": "jpg"
            }

            output = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input=input_data
            )

            output_path = f"temp/{img_id}_output_{i+1}.jpg"
            with open(output_path, "wb") as f:
                output_bytes = output.read()
                f.write(output_bytes)

            image_url = f"{API_BASE_URL}/temp/{img_id}_output_{i+1}.jpg"
            urls.append(image_url)
            time.sleep(0.3)  # Pequeno delay para evitar sobrecarga

        return {"image_urls": urls}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

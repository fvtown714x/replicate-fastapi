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
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

@app.post("/gerar-headshot")
async def gerar_headshot(
    image: UploadFile = File(...),
    clothing: str = Form(...),
    background: str = Form(...),
):
    try:
        # Salvar imagem temporária
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_path = f"temp/{img_id}.{file_ext}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Prompt para a Replicate
        prompt = f"Professional headshot, wearing {clothing}, background: {background}"

        input_data = {
            "prompt": prompt,
            "input_image": open(input_path, "rb"),
            "output_format": "jpg"
        }

        output = replicate.run(
            "black-forest-labs/flux-kontext-pro",
            input=input_data
        )

        # Salvar imagem gerada
        output_path = f"temp/{img_id}_output.jpg"
        with open(output_path, "wb") as f:
            output_bytes = output.read()
            with open(output_path, "wb") as f:
                f.write(output_bytes)


        # Retornar URL acessível publicamente
        image_url = f"http://127.0.0.1:8000/temp/{img_id}_output.jpg"
        time.sleep(0.3)  # Pequena espera (300ms)
        return {"image_url": image_url}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})
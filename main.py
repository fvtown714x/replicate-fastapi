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
import traceback

load_dotenv()

app = FastAPI()

API_BASE_URL = "https://replicate-fastapi.onrender.com"

# CORS (ajuste aqui para seu domínio Webflow)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste para ["https://jobodega.webflow.io"] em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cria pasta para arquivos temporários
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

@app.post("/gerar-headshot")
async def gerar_headshot(
    image: UploadFile = File(...),
    clothing: str = Form(...),       # Deve ser uma string JSON com lista
    background: str = Form(...),     # Deve ser uma string JSON com lista
    profession: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...)
):
    try:
        # Parse as listas enviadas como JSON
        clothing_list = json.loads(clothing)
        background_list = json.loads(background)

        if not clothing_list or not background_list:
            return JSONResponse(status_code=400, content={"erro": "clothing ou background vazio."})

        # Salvar imagem original temporária
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_path = f"temp/{img_id}.{file_ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        urls = []

        # Loop para gerar combinações (mínimo do tamanho das duas listas)
            # Loop para gerar combinações (mínimo do tamanho das duas listas)
        for idx, (clothe, bg) in enumerate(zip(clothing_list, background_list)):
            prompt = (
                f"Professional LinkedIn headshot of a {age}-year-old {gender.lower()} {profession}, "
                f"wearing {clothe}, with a background of {bg}. "
                f"High-quality DSLR photo, studio lighting, shallow depth of field, realistic details, professional attire, clean background."
            )
            print(f"🔹 Prompt {idx+1}: {prompt}")
        
            with open(input_path, "rb") as image_file:
                output = replicate.run(
                    "black-forest-labs/flux-kontext-pro",
                    input={
                        "prompt": prompt,
                        "input_image": image_file,
                        "output_format": "jpg"
                    }
                )
        
            # Salvar imagem gerada
            output_path = f"temp/{img_id}_output_{idx+1}.jpg"
            with open(output_path, "wb") as f:
                f.write(output.read())
        
            image_url = f"{API_BASE_URL}/temp/{img_id}_output_{idx+1}.jpg"
            urls.append(image_url)
            time.sleep(0.3)



        return {"image_urls": urls}

    except Exception as e:
        print("❌ Erro:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"erro": str(e)})

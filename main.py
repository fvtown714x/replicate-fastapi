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

def map_attire_description(attire, gender):
    attire = attire.lower()
    gender = gender.lower()
    if attire == "business professional":
        return "a suit and tie" if gender == "male" else "a professional corporate outfit"
    elif attire == "business casual":
        return "a suit with no tie" if gender == "male" else "a professional corporate outfit"
    elif attire == "casual":
        return "a t-shirt, a button-up shirt, a flannel, a sweater vest, or something currently trendy and fashionable"
    elif attire == "medical":
        return "modern medical attire, wearing a clean white lab coat over scrubs or business-casual medical clothing, with a stethoscope around their neck and an ID badge clipped to their coat"
    elif attire == "scientist":
        return "a white lab coat over casual-professional clothes, safety goggles on their head or eyes, and blue nitrile gloves"
    else:
        return attire

def map_background_description(background):
    background = background.lower()
    if background == "light gray":
        return "a neutral light grey photo studio background"
    elif background == "soft gradient":
        return "a soft gradient background"
    elif background == "corporate office":
        return "a bright and modern office with desks and computers in the background"
    elif background == "natural outdoors":
        return "an open space within a famous US National Park with natural daytime lighting"
    elif background == "trendy indoor space":
        return "a picturesque view of the inside of a world-famous tourist attraction"
    elif background == "startup office":
        return "a modern startup office background"
    else:
        return background

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
        for idx, (clothe, bg) in enumerate(zip(clothing_list, background_list)):
            attire_desc = map_attire_description(clothe, gender)
            background_desc = map_background_description(bg)
        
            prompt = (
                f"Create a professional headshot of this subject in soft studio lighting, "
                f"wearing {attire_desc} outfit, background is {background_desc}. "
                f"Maintain precise replication of subject's pose, head tilt, and eye line, angle toward the camera, "
                f"skin tone, and any jewelry."
            )
        
            print(f"🔹 Prompt {idx+1}: {prompt}")
        
            with open(input_path, "rb") as image_file:
                output = replicate.run(
                    "black-forest-labs/flux-kontext-pro",
                    input_data={
                        "prompt": prompt,
                        "input_image": image_file,
                        "output_format": "jpg"
                    }
                )
        
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

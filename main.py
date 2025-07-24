from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import replicate
import os
import time
import shutil
from dotenv import load_dotenv
from uuid import uuid4
import requests

load_dotenv()

app = FastAPI()

API_BASE_URL = "https://replicate-fastapi.onrender.com"

# CORS para aceitar Webflow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cria pasta e serve imagens em /temp
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")


@app.post("/gerar-headshot")
async def gerar_headshot(
    image: UploadFile = File(...),
    gender: str = Form(...),
    profession: str = Form(...),
    age: str = Form(...),
    clothing: str = Form(...),
    background: List[str] = Form(...),
):
    try:
        # Salva imagem recebida localmente
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_filename = f"{img_id}.{file_ext}"
        input_path = os.path.join("temp", input_filename)

        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Cria URL pública da imagem que acabou de salvar
        image_input_url = f"{API_BASE_URL}/temp/{input_filename}"

        image_urls = []
        selected_backgrounds = background[:5]

        for i, bg in enumerate(selected_backgrounds):
            prompt = f"""
            Provide an ultra high-resolution professional headshot for this
            {gender}, 
            who works as a {profession},
            aged {age},
            give him a {clothing} outfit. 
            The background of this image is {bg} and should be related to the {profession}. 
            The image is shot with studio-quality lighting, shallow depth of field, and soft shadows.  
            The shots should have a professional look for corporate use. DSLR photo style, centered composition, no accessories unless specified.
            """

            # ✅ Agora usando URL como input
            prediction = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input={
                    "prompt": prompt,
                    "image": image_input_url,
                    "output_format": "jpg"
                }
            )

            if isinstance(prediction, list):
                image_url_replicate = prediction[0]
            else:
                image_url_replicate = prediction

            # Faz download da imagem gerada
            img_data = requests.get(image_url_replicate).content
            output_path = f"temp/{img_id}_{i}_output.jpg"
            with open(output_path, "wb") as f:
                f.write(img_data)

            # Adiciona URL final que será servida pelo Render
            image_url = f"{API_BASE_URL}/temp/{img_id}_{i}_output.jpg"
            image_urls.append(image_url)

            time.sleep(0.5)  # reduz carga no servidor da Replicate

        return {"image_urls": image_urls}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"erro": str(e)})

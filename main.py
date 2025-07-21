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

load_dotenv()

app = FastAPI()

# URL base para montar os links das imagens
API_BASE_URL = "https://replicate-fastapi.onrender.com"

# CORS configurado para aceitar Webflow
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://jobodega.webflow.io",
        "https://www.jobodega.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cria e serve os arquivos gerados no endpoint /temp
os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")


@app.post("/gerar-headshot")
async def gerar_headshot(
    image: UploadFile = File(...),
    gender: str = Form(...),
    profession: str = Form(...),
    age: str = Form(...),
    clothing: str = Form(...),
    background: List[str] = Form(...),  # múltiplos valores possíveis
):
    try:
        # Salva imagem temporariamente
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_filename = f"{img_id}.{file_ext}"
        input_path = os.path.join("temp", input_filename)

        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Monta a URL pública para a imagem
        image_url_input = f"{API_BASE_URL}/temp/{input_filename}"

        # Gera até 5 imagens com backgrounds diferentes
        selected_backgrounds = background[:5]
        image_urls = []

        for i, bg in enumerate(selected_backgrounds):
            prompt = f"""
            Provide an ultra high-resolution professional headshot of a
            {gender}, 
            who works as a {profession},
            aged {age},
            wearing {clothing}. 
            The background of this image is {bg} and should be related to the {profession}. 
            The image is shot with studio-quality lighting, shallow depth of field, and soft shadows.  
            The shots should have a professional look for corporate use. DSLR photo style, centered composition, no accessories unless specified.
            """

            # Executa o modelo com a imagem da URL
            output = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input={
                    "prompt": prompt,
                    "input_image": image_url_input,
                    "output_format": "jpg"
                }
            )

            # Salva a imagem de saída
            output_path = f"temp/{img_id}_{i}_output.jpg"
            with open(output_path, "wb") as f:
                f.write(output.read())

            # Monta a URL pública da imagem gerada
            image_url = f"{API_BASE_URL}/temp/{img_id}_{i}_output.jpg"
            image_urls.append(image_url)

            time.sleep(0.5)  # delay leve por segurança

        return {"image_urls": image_urls}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

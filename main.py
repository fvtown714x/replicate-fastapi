from typing import List
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substitua por Webflow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp", exist_ok=True)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")


@app.post("/gerar-headshot")
async def gerar_headshot(
    image: UploadFile = File(...),
    gender: str = Form(...),
    age: str = Form(...),
    profession: str = Form(...),
    clothing: str = Form(...),
    backgrounds: List[str] = Form(...)
):
    try:
        # Salvar imagem temporária
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_path = f"temp/{img_id}.{file_ext}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Juntar os backgrounds múltiplos
        background_str = ", ".join(backgrounds)

        # Criar o prompt
        prompt = f"""
        Provide an ultra high-resolution professional LinkedIn headshot of a
        {gender}, who works as a {profession}, aged {age}, wearing {clothing}, looking confident and approachable.
        The background of this image is {background_str} and should be related to the {profession}.
        The image is shot with studio-quality lighting, shallow depth of field, crisp facial detail, and soft shadows.
        The person should have eye contact with the camera and a subtle smile.
        The shots should have a professional look for corporate use.
        DSLR photo style, centered composition, no accessories unless specified.
        """

        urls = []

        for i in range(5):  # Gerar 5 imagens
            input_data = {
                "prompt": prompt.strip(),
                "input_image": open(input_path, "rb"),
                "output_format": "jpg"
            }

            output = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input=input_data
            )

            # Salvar cada imagem
            output_path = f"temp/{img_id}_{i}_output.jpg"
            with open(output_path, "wb") as f:
                output_bytes = output.read()
                f.write(output_bytes)

            urls.append(f"{API_BASE_URL}/temp/{img_id}_{i}_output.jpg")
            time.sleep(0.3)

        return {"image_urls": urls}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

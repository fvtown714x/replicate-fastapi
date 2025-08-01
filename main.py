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
    name: str = Form(...),
    gender: str = Form(...),
    profession: str = Form(...),
    age: str = Form(...),
    clothing: List[str] = Form(...),
    background: List[str] = Form(...),
):
    try:
        # Salva imagem original
        file_ext = image.filename.split(".")[-1]
        img_id = str(uuid4())
        input_filename = f"{img_id}.{file_ext}"
        input_path = os.path.join("temp", input_filename)

        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        image_input_url = f"{API_BASE_URL}/temp/{input_filename}"
        image_urls = []

        selected_styles = list(zip(clothing[:5], background[:5]))

        for i, (cloth, bg) in enumerate(selected_styles):
            prompt = f"""
            Professional DSLR headshot of a {gender} {profession}, around {age} years old, named {name}, wearing {cloth}, with a {bg} background.
            Studio lighting, shallow depth of field, ultra realistic.
            """

            prediction = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input={
                    "prompt": prompt.strip(),
                    "image": image_input_url,
                    "output_format": "jpg"
                }
            )

            image_url_replicate = prediction[0] if isinstance(prediction, list) else prediction
            img_data = requests.get(image_url_replicate).content

            output_path = f"temp/{img_id}_{i}_output.jpg"
            with open(output_path, "wb") as f:
                f.write(img_data)

            final_url = f"{API_BASE_URL}/temp/{img_id}_{i}_output.jpg"
            image_urls.append(final_url)

            time.sleep(0.5)

        return {"image_urls": image_urls}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"erro": str(e)})

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from io import BytesIO
import logging
import uvicorn

app = FastAPI(title="Image Inference API", description="API for running inference on images to detect fire or smoke.", version="1.0")


logging.basicConfig(level=logging.INFO)
# Load models and processors
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# logging.info(f"Using device: {DEVICE}")
smolvlm_model_name = "HuggingFaceTB/SmolVLM-Instruct"
smolvlm_processor = AutoProcessor.from_pretrained(smolvlm_model_name)
smolvlm_model = AutoModelForVision2Seq.from_pretrained(smolvlm_model_name, _attn_implementation="eager").to(DEVICE)

moondream_model_name = "vikhyatk/moondream2"
moondream_model = AutoModelForCausalLM.from_pretrained(moondream_model_name, revision="2025-01-09", trust_remote_code=True, device_map={"": DEVICE})
# Helper function to load image
def load_image(image_file):
    return Image.open(image_file)

# Endpoint for SmolVLM model
@app.post("/predict/smolvlm")
async def predict_smolvlm(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    prompt = smolvlm_processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "You are helpful in identifying fire or smoke in images. Answer in English only and answer in a single word, either 'yes' or 'no'.Do you see fire or smoke in the image?"}]}],
        add_generation_prompt=True
    )
    inputs = smolvlm_processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    generated_ids = smolvlm_model.generate(**inputs, max_new_tokens=500)
    generated_texts = smolvlm_processor.batch_decode(generated_ids, skip_special_tokens=True)
    prediction = generated_texts[0].split("Assistant:")[1].strip().lower()
    return {"prediction": prediction}

# Endpoint for Moondream model
@app.post("/predict/moondream")
async def predict_moondream(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    try:
        enc_image = moondream_model.encode_image(image)
        prediction_text = moondream_model.query(enc_image, "You are helpful in identifying fire or smoke in images. Answer in English only and answer in a single word, either 'yes' or 'no'.Do you see fire or smoke in the image?")['answer'].lower()
        return {"prediction": prediction_text}
    except Exception as e:
        return {"error": str(e)}

# Run the app with: uvicorn vlm_fire:app --reload

if __name__ == "__main__":
    logging.info("***************************** Starting app *****************************")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
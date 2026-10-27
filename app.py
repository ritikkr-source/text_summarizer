from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.responses import HTMLResponse
from pathlib import Path
import torch
import re

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"
HTML_FILE = BASE_DIR / "index.html"

# Select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))

model.to(device)
model.eval()


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n|\n|\r", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    return text.strip()


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    if not dialogue:
        raise ValueError("Please enter some text to summarize.")

    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def home():
    if not HTML_FILE.exists():
        return HTMLResponse(
            content="index.html was not found.",
            status_code=404
        )

    return HTMLResponse(
        content=HTML_FILE.read_text(encoding="utf-8")
    )


# Summarization endpoint (supports both URL formats)
@app.post("/summarize")
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

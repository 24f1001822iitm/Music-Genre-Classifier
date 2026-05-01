import gradio as gr
import torch
import librosa
import numpy as np
from transformers import ASTForAudioClassification, ASTFeatureExtractor

MODEL_ID = "chopadaansh/ast-music-genre-classifier"
BASE_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000

LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

print("Loading model...")
model = ASTForAudioClassification.from_pretrained(MODEL_ID)
feature_extractor = ASTFeatureExtractor.from_pretrained(BASE_MODEL_ID)
model.eval()
print("Model ready.")


def classify(audio_path):
    if audio_path is None:
        return {g: 0.0 for g in LABELS}

    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    inputs = feature_extractor(
        waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    return {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(type="filepath", label="Upload a music clip"),
    outputs=gr.Label(num_top_classes=5, label="Predicted Genre"),
    title="Music Genre Classifier",
    description="Upload a music clip to classify its genre. AST model fine-tuned on GTZAN (10 genres).",
)

if __name__ == "__main__":
    demo.launch()

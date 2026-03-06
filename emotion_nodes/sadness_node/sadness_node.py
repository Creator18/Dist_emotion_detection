from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Initialize the FastAPI app
app = FastAPI()

# Paths to the saved files
m_path = r"sadness_classification_model.pth"
t_path = "/app/tokenizer"

print(f"Tokenizer path: {t_path}")


# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(t_path)
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.load_state_dict(torch.load(m_path, map_location=device))
model.to(device)
model.eval()

# Request body format
class PredictionRequest(BaseModel):
    text: str

# Define prediction route
@app.post("/predict")
def predict_emotion(request: PredictionRequest):
    try:
        # Tokenize input text
        encoding = tokenizer.encode_plus(
            request.text,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Move tensors to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            confidence_scores = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = confidence_scores[0, predicted_class].item()

        return {
            "prediction": "Sadness" if predicted_class == 1 else "Not Sadness",
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

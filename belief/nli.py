from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_nli_model():
    return AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli").to(device)

def load_nli_tokenizer():
    return AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

def run_nli(premise, hypothesis, model, tokenizer):
    input_ids = tokenizer.encode(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    output = model(input_ids)

    prediction = torch.softmax(output.logits[0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) , 4) for pred, name in zip(prediction, label_names)}

    return prediction
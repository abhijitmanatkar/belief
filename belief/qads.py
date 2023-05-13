"""
Convert a Question + Answer pair to a Declarative Sentence
QADS = Question Answer -> Declarative Sentence
"""

import torch
from transformers import T5ForConditionalGeneration, T5Config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_qads_model(weights_path):
    weights = torch.load(weights_path)['model']
    config = T5Config.from_pretrained('t5-base')
    qads = T5ForConditionalGeneration(config).to(device)
    qads.load_state_dict(weights)
    
    return qads


def run_qads(qas, model, tokenizer):
    """
        qas: List of q,a tuples
    """
    
    inpstrs = [q + " " + a for (q,a) in qas]
    inpids = tokenizer(inpstrs, truncation=True, padding=True, return_tensors="pt").input_ids.to(device)
    
    out = qads.generate(inpids, max_length=500)
    out_text = tokenizer.batch_decode(out, skip_special_tokens=True)
    
    return out_text
    
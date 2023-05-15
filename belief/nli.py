from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

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

def pairwise_nli_precompute(facts, model, tokenizer, chunk_size=50):
    
    nli_outs = []
    
    with tqdm(total=len(facts) * (len(facts) - 1), leave=True, position=0, ncols=80) as pbar:
    
        for fact1 in facts:

            i = 0

            while chunk_size * i < len(facts):
                
                src_sents = []
                dest_sents = []
                
                input_l1 = []
                input_l2 = []

                for fact2 in facts[chunk_size*i : min(len(facts), chunk_size * (i+1))]:

                    if fact1.sentence == fact2.sentence:
                        continue
                    
                    src_sents.append(fact1.sentence)
                    dest_sents.append(fact2.sentence)
                    input_l1.append(fact1.get_assertion())
                    input_l2.append(fact2.get_assertion())

                input_ids = tokenizer(input_l1, input_l2, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
                output = model(input_ids)
                prediction = torch.softmax(output.logits, -1).tolist()

                del input_ids, output
                torch.cuda.empty_cache()

                for src, dest, scores in zip(src_sents, dest_sents, prediction):
                    nli_outs.append((
                        src,
                        dest,
                        {'entailment': scores[0], 'neutral': scores[1], 'contradiction': scores[2]}
                    ))

                pbar.update(min(len(facts), chunk_size * (i+1)) - chunk_size*i)
                i += 1
    
    pbar.close()
    
    return nli_outs

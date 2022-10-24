import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import string
import json
import random
from belief.lmbb import Proposition, Predicate

def get_scores(model, tokenizer, raw_input, output_options):
    output_scores = []
    with torch.no_grad():
        input_ids = tokenizer.encode(raw_input, return_tensors="pt")
        for option in output_options:
            output_ids = tokenizer.encode(option, return_tensors="pt")
            res = model(input_ids, labels=output_ids, return_dict=True)
            score = torch.exp(-res.loss)
            output_scores.append((option, score.numpy()))
    return output_scores

def get_raw_input(question, options):
    raw_input = question + ' \n'
    for option_num, option in zip(list(string.ascii_uppercase), options):
        raw_input += f' ({option_num}) {option}'
    return raw_input

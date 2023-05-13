import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import string
import json
import random
from belief.macaw_utils import decompose_slots, compute_answer, run_model_with_outputs


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# def get_scores(model, tokenizer, raw_input, output_options):
#     output_scores = []
#     with torch.no_grad():
#         input_ids = tokenizer.encode(raw_input, return_tensors="pt")
#         for option in output_options:
#             output_ids = tokenizer.encode(option, return_tensors="pt")
#             res = model(input_ids, labels=output_ids, return_dict=True)
#             score = torch.exp(-res.loss)
#             output_scores.append((option, score.numpy()))
#     return output_scores

# def get_raw_input(question, options):
#     raw_input = question + ' \n'
#     for option_num, option in zip(list(string.ascii_uppercase), options):
#         raw_input += f' ({option_num}) {option}'
#     return raw_input


with open('belief/non_countable.txt', 'r') as f:
    uncountables = f.read().split('\n')


def noun_fluenterer(noun, relation=None):
    """
    Make a noun phrase 'fluenter' (more fluent) before putting it in a
    template.  note we only a.) check if the noun is in a list of known
    non-countables or has a relation with a certain type, and b.) look at the
    first letter of the input to decide whether to put a or an.

    :param noun: the noun (phrase) -- subject or object -- to make more fluent
    :param relation: BeliefBank relation
    :return: a string with the prettified noun phrase
    """

    if noun in uncountables:
        return noun

    if relation is not None:
        if relation in ['CapableOf', 'MadeOf', 'HasProperty']:
            return noun

    if noun[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an ' + noun

    return 'a ' + noun

##################################
# Macaw basic
##################################

def load_macaw():
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large", load_in_8bit=True, device_map='auto')
    return model

def load_tokenizer():
    return AutoTokenizer.from_pretrained("allenai/macaw-large")

def macaw_input(question="", answer="", options=(), explanation="", context="", targets='AE'):
    
    if len(question) > 0:
        question_str = '$question$ = ' + question  + " ; "
    elif 'Q' in targets:
        question_str = "$question$ ; "
    
    if len(explanation) > 0:
        explanation_str = "$explanation$ = " + explanation + " ; "
    elif 'E' in targets:
        explanation_str = "$explanation$ ; "
    else:
        explanation_str = ""
    
    if len(answer) > 0:
        answer_str = "$answer$ = " + answer 
        if len(context) > 0:
            answer_str += " ; "
    elif 'A' in targets:
        answer_str = "$answer$"
        if len(context) > 0:
            answer_str += " ; "
    else:
        answer_str = ""
    
    if len(context) > 0:
        context_str = "$context$ = " + context
    else:
        context_str = ""
    
    letters = list(string.ascii_uppercase)
    if len(options) > 0:
        option_str = "$mcoptions$ = "
        for letter, option in zip(letters, options):
            option_str += f"({letter}) {option} "
        option_str += "; "
    elif 'M' in targets:
        option_str = "$mcoptions$ ; "
    else:
        option_str = ""
    
    return f"{question_str}{explanation_str}{option_str}{answer_str}{context_str}"

def run_macaw(input_str, model, tokenizer):
    input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)
    outs = model.generate(input_ids, max_length=500, early_stopping=True)
    return tokenizer.batch_decode(outs, skip_special_tokens=True)[0]

def get_macaw_outs(input_str, model, tokenizer):
    output_str = run_macaw(input_str, model, tokenizer)
    slots = decompose_slots(output_str)
    for slot in slots:
        if '~AND~' in slots[slot]:
            slots[slot] = slots[slot].split('~AND~')[0].strip()
    return slots
    
##################################
# What Do I Know?
##################################

def get_questions(entity):

    entity = noun_fluenterer(entity)
    
    what_is = f"What is {entity}?"
    made_of = f"What is {entity} made of?"
    capable_of = f"What is {entity} capable of?"
    has_what_part = f"What parts does {entity} have?"
    has_what_property = f"What properties does {entity} have?"
    which_category = f"Which category does {entity} belong to?"

    return [what_is, made_of, capable_of, has_what_part, has_what_property, which_category]

def get_qa_pairs(entity, model, tokenizer):
    
    questions = get_questions(entity)

    inpstrs = [macaw_input(targets='AE', question=question) for question in questions]
    inpids = tokenizer(inpstrs, truncation=True, padding=True, return_tensors="pt").input_ids.to(device)
    
    num_beams = 3
    num_return_sequences = 3
    
    out = model.generate(
        input_ids=inpids, 
        max_length=500,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True
    )
    
    out_text = tokenizer.batch_decode(out, skip_special_tokens=True)
    
    qas = set()
    for i,q in enumerate(questions):
        for j in range(3*i, 3*(i+1)):
            out_str = out_text[j]
            slots = decompose_slots(out_str)
            ans = slots['answer'] if 'answer' in slots else ''
            qas.add((q,ans))
    
    return list(qas)


##################################
# Macaw Scoring
##################################

def get_output_strings_from_options(options):
    """
    Formats a list of options intoo output_str as required by macaw
    """
    l = []
    for option in options:
        l.append((f"$answer$ = {option}", option))
    return l

def get_macaw_scores(inp_str, options, model, tokenizer):
    out_str = get_output_strings_from_options(options)
    res = run_model_with_outputs(model, tokenizer, device, inp_str, out_str)
    scores = {}
    for r in res:
        scores[r['output_text']] = r['score']
    return scores

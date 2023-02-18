import random
from belief.lmbb import LMBB, Proposition, Predicate
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import torch

def load_facts(filepath, num_batches=10):
    with open(filepath, 'r') as f:
        fact_data = json.load(f)

    facts = []
    for subject in fact_data.keys():
        for predicate_str_rep in fact_data[subject]:
            prop = Proposition(
                subject=subject,
                predicate=Predicate(predicate_str_rep),
                boolean=True if fact_data[subject][predicate_str_rep] == "yes" else False
            )
            facts.append(prop)
    random.shuffle(facts)
    
    batch_size = len(facts) // num_batches
    batches = [facts[batch_size*i:batch_size*(i+1)] for i in range(num_batches)]
    return batches


def build_bb(
    model_name="allenai/unifiedqa-v2-t5-base-1251000",
    facts_file="./data/silver_facts.json", 
    constraints_file="./data/constraints_v2.json",
    num_batches=10,
    constraint_solving=True,
    with_feedback=True,
    forward_weight=None, 
    backward_weight=None, 
):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    with open(constraints_file, 'r') as f:
        constraint_data = json.load(f)
    
    lmbb = LMBB(
        model=model,
        tokenizer=tokenizer,
        raw_constraints=constraint_data['links'],
        forward_weight=forward_weight,
        backward_weight=backward_weight
    )
    
    fact_batches = load_facts(facts_file, num_batches)
    
    consistency_list = []
    f1_list = []
    fact_props = []
    
    for i in range(num_batches):
        lmbb.add_belief_set(fact_batches[i], constraint_solving, with_feedback)
        fact_props += fact_batches[i]
        c = lmbb.calculate_consistency()
        f1 = lmbb.calculate_f1(fact_props)
        print(f"Batch {i+1} : F1 = {f1}, consistency = {c}")
        consistency_list.append(c)
        f1_list.append(f1)
        
    return lmbb, consistency_list, f1_list

def calibrate(
    forward_weights=[], 
    backward_weights=[],
    model_name="allenai/unifiedqa-v2-t5-base-1251000",
    facts_file="./data/calibration_facts.json", 
    constraints_file="./data/constraints_v2.json",
    num_batches=10,
    constraint_solving=True,
    with_feedback=True, 
):
    eval_dict = {}
    for fw in forward_weights:
        for bw in backward_weights:
            
            print(f"Forward weight: {fw}, Backward_weight: {bw}")
            _, consistency_list, f1_list = build_bb(
                model_name, 
                facts_file, 
                constraints_file,
                num_batches,
                constraint_solving, 
                with_feedback, fw, bw)
            
            eval_dict[(fw, bw)] = (consistency_list[-1], f1_list[-1])
            print(f"======= Final eval ========")
            print(f"(fw={fw}, bw={bw}) => (f1={f1_list[-1]}), consistency={consistency_list[-1]}")
            print()
    return eval_dict
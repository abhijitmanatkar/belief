from belief.evaluation import load_facts
from belief.utils import get_macaw_scores, macaw_input


def get_raw_outputs(facts, model, tokenizer):
    """
    Returns raw macaw model outputs on facts
    """
    raw_outputs = []
    
    options = ["yes", "no"]
    
    for prop in facts:
        question = prop.get_question()
        input_str = macaw_input(question=question, options=options, targets='A')
        scores = get_macaw_scores(input_str, options, model, tokenizer)
        raw_outputs.append((prop, scores))
        
    return raw_outputs
    
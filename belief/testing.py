import random
from belief.lmbb import Proposition, Predicate


def load_facts(filepath):
    with open(filepath, 'r') as f:
        fact_data = json.load(f)

    facts = []
    for subject in fact_data.keys():
        for predicate_str_rep in fact_data[subject]:
            prop = Proposition(
                subject=subject,
                predicate=Predicate(predicate_str_rep)
            )
            facts.append(prop)
    random.shuffle(facts)
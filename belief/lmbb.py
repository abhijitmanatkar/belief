from z3 import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import torch
import string
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from typing import List, Dict
import copy
import json
import logging
import random
from belief.utils import get_raw_input, get_scores

WEIGHT_PRECISION = 3

class FeedbackType(Enum):
    RELEVANT = 0
    ON_TOPIC_RANDOM = 1

class Predicate():
    """A predicate is a generalized sentence which can be substituted with entities.
    Example 'IsA,dog' is a predicate in which if we substitute the entity 'poodle',
    we get the sentence 'A poodle is a dog'.

    Args:
        str_rep: String representation of predicate, ex 'IsA,mammal', 'HasA,tail'
    """

    def __init__(self, str_rep):
        self.str_rep = str_rep
        self.relation, self.object = str_rep.split(",")
        if self.relation not in ['CapableOf', 'HasA', 'HasPart', 'HasProperty', 'IsA', 'MadeOf']:
            raise Exception(f"{self.relation} is an unsupported relation")
    
    def __repr__(self):
        return f"{self.relation},{self.object}"

    def substitute(self, subject):
        """Returns sentence, i.e. 'subject,relation,object' on substituting subject in
        the predicate.
        """
        return f"{subject},{self.relation},{self.object}"

class Proposition():
    """A proposition, Can be a belief when present in a belief bank

    Attributes:
        subject: The subject of the proposition
        predicate: The general predicate of the proposition in which the subject is substituted.
        boolean: The Truth value of the sentence
        weight: Optional attribute indicating the weight of the proposition. 
    """

    def __init__(
        self,
        subject: str,
        predicate: Predicate,
        boolean: bool = True,
        weight: float = -99999,
    ):
        self.subject = subject
        self.predicate = predicate
        self.boolean = boolean
        self.weight = int(round(weight * (10**WEIGHT_PRECISION), WEIGHT_PRECISION))

    def __repr__(self):
        return f"({self.sentence}, {self.boolean}, {self.weight / (10**WEIGHT_PRECISION)})"

    @property
    def sentence(self):
        return self.predicate.substitute(self.subject)

    def get_nl_sentence(self):
        """Returns natural language sentence expressing the proposition"""
        if self.predicate.relation == 'IsA':
            return f"{self.subject} is a {self.predicate.object}."
        if self.predicate.relation == "MadeOf":
            return f"{self.subject} is made of {self.predicate.object}."
        if self.predicate.relation == "CapableOf":
            return f"{self.subject} is capable of {self.predicate.object}."
        if self.predicate.relation == "HasA":
            return f"{self.subject} has a {self.predicate.object}."
        if self.predicate.relation == "HasPart":
            return f"A {self.predicate.object} is part of a {self.subject}."
        if self.predicate.relation == "HasProperty":
            return f"{self.subject} has the property of being {self.object}."

    def get_nl_question(self):
        """Returns natural language sentence asking the proposition"""
        if self.predicate.relation == 'IsA':
            return f"Is a {self.subject} a {self.predicate.object}?"
        if self.predicate.relation == "MadeOf":
            return f"Is a {self.subject} made of {self.predicate.object}?"
        if self.predicate.relation == "CapableOf":
            return f"Is a {self.subject} capable of {self.predicate.object}?"
        if self.predicate.relation == "HasA":
            return f"Does a {self.subject} have a {self.predicate.object}?"
        if self.predicate.relation == "HasPart":
            return f"Is a {self.predicate.object} part of a {self.subject}?"
        if self.predicate.relation == "HasProperty":
            return f"Does {self.subject} have the property of being {self.object}?"

class Constraint():
    """Class for constraint
    
    Attributes:
        src_predicate: The source predicate
        dest_predicate: The destination predicate
        weight: Penalty for violation of constraint
        implication: Indicates the type of implication in the constraint (T->F, T->T, F->T, F->F)
    """

    def __init__(
        self,
        src_predicate: Predicate,
        dest_predicate: Predicate,
        weight: float,
        implication: str
    ):
        self.src_predicate = src_predicate
        self.dest_predicate = dest_predicate
        self.weight = int(round(weight * (10**WEIGHT_PRECISION), WEIGHT_PRECISION))
        self.implication = implication

    @classmethod
    def from_raw(cls, raw_link):
        if raw_link['direction'] == 'forward':
            src_predicate = raw_link['source']
            dest_predicate = raw_link['target']
        elif raw_link['direction'] == 'back':
            src_predicate = raw_link['target']
            dest_predicate = raw_link['source']

        return Constraint(
            src_predicate=Predicate(src_predicate),
            dest_predicate=Predicate(dest_predicate),
            weight=raw_link['score'],
            implication=raw_link['weight']
        )

    def __repr__(self):
        s, d = self.implication.split('_')
        return f"Constraint(<{'¬' if s == 'no' else ''}{self.src_predicate.str_rep} ⟶ {'¬' if d == 'no' else ''}{self.dest_predicate.str_rep}>, {self.weight / {10**WEIGHT_PRECISION}})"


class LMBB():
    """Langauge Model + Belief Bank"""

    def __init__(self, model, tokenizer, raw_constraints):
        self.model = model
        self.tokenizer = tokenizer
        
        self.constraints = [Constraint.from_raw(c) for c in raw_constraints]
        self.links: Dict[str, List[Constraint]] = defaultdict(list)
        for constraint in self.constraints:
            self.links[constraint.src_predicate.str_rep].append(constraint)
        
        # Dictionary where the key is 'subject,relation,predicate' and value is a proposition
        self.beliefs = dict()

        # Feedback config
        self.num_random_on_topic_beliefs = 3
        self.num_relevant_beliefs = 3

    def get_beliefs_by_subject(self, subject):
        subject_beliefs = [self.beliefs[key] for key in self.beliefs.keys() if key.split(",")[0] == subject]
        return subject_beliefs

    def add_belief(
        self, 
        proposition: Proposition,
        constraint_solving: bool = True,
        with_feedback: bool = True,
        feedback_type: FeedbackType = FeedbackType.RELEVANT
    ):
        """Adds a single proposition to the systems existing set of beliefs
        Args:
            proposition: New proposition to add
            constraint_solving: Whether to use constraint solving or not
            with_feedback: Whether to use feedback or not
            feedback_type: RELEVANT / ON_TOPIC_RANDOM
        """
        
        # Get proposition after passing through language model with/without feedback
        proposition = self.query(proposition, with_feedback, feedback_type)

        # Optional constraint solving
        if constraint_solving:
            beliefs = list(self.beliefs.values())
            beliefs.append(proposition)
            new_beliefs = self.maxSat(beliefs)
        else:
            new_beliefs = [proposition]
        self.set_beliefs(new_beliefs)

    def add_belief_set(
        self, 
        propositions: List[Proposition],
        constraint_solving: bool = True,
        with_feedback: bool = True,
        feedback_type: FeedbackType = FeedbackType.RELEVANT
    ):
        """Adds a set of proposition to the systems existing set of beliefs
        Args:
            propositions: List of new propositions to add
            constraint_solving: Whether to use constraint solving or not
            with_feedback: Whether to use feedback or not
            feedback_type: RELEVANT / ON_TOPIC_RANDOM
        """
        propositions = [self.query(proposition, with_feedback, feedback_type) for proposition in propositions]
        
        # Optional constraint solving
        if constraint_solving:
            beliefs = list(self.beliefs.values())
            beliefs += propositions
            new_beliefs = self.maxSat(beliefs)
        else:
            new_beliefs = propositions
        self.set_beliefs(new_beliefs)

    def set_beliefs(self, new_beliefs):
        for belief in new_beliefs:
            self.beliefs[belief.sentence] = belief

    def query(self, proposition: Proposition, with_feedback=True, feedback_type=FeedbackType.RELEVANT):
        """
        Args:
            proposition: the query to the LMBB system
            with_feedback: Whether to use context from existing beliefs or not
            feedback_type: RELEVANT / ON_TOPIC_RANDOM 

        Returns:
            proposition: The updated proposition with truth value and weight
        """
        
        feedback_beliefs = []
        if with_feedback:
            feedback_beliefs = self.feedback(proposition, feedback_type)
        feedback_string = " ".join([belief.get_nl_sentence() for belief in feedback_beliefs])
        question = feedback_string + ' ' + proposition.get_nl_question()

        # TODO: Grammar model to refine the question
        
        options = ['yes', 'no']
        
        raw_input = get_raw_input(question, options)
        logging.debug(raw_input)
        scores = get_scores(self.model, self.tokenizer, raw_input, options)
        logging.debug(scores)
        answer = max(scores, key=lambda x: x[1])

        return Proposition(
            subject=proposition.subject,
            predicate=proposition.predicate,
            boolean=True if answer[0] == 'yes' else False,
            weight=answer[1]
        )
    
    def maxSat(self, beliefs: List[Proposition]):
        """Run MaxSAT solver considering 'beliefs' and constraints
        
        Args:
            beliefs: List of propositions
        
        Returns:
            new_beliefs: List of modified beliefs to ensure maximum satisfiability
        """
        optim = Optimize()

        belief_bools = {}
        belief_props = {}
        
        subjects = set()

        for prop in beliefs:
            subjects.add(prop.subject)
            sentence = prop.sentence
            if sentence not in belief_bools:
                belief_bools[sentence] = Bool(sentence)
                belief_props[sentence] = prop
            if prop.boolean == True:
                optim.add_soft(belief_bools[sentence], prop.weight)
            else:
                optim.add_soft(Not(belief_bools[sentence]), prop.weight)

        for constraint in self.constraints:
            for subject in subjects:
                src_sent = constraint.src_predicate.substitute(subject)
                dest_sent = constraint.dest_predicate.substitute(subject)

                src_bool = belief_bools[src_sent] if src_sent in belief_bools else Bool(src_sent)
                dest_bool = belief_bools[dest_sent] if dest_sent in belief_bools else Bool(dest_sent)

                if constraint.implication == "yes_yes":
                    optim.add_soft(Implies(src_bool, dest_bool), constraint.weight)
                elif constraint.implication == "yes_no":
                    optim.add_soft(Implies(src_bool, Not(dest_bool)), constraint.weight)
                elif constraint.implication == "no_yes":
                    optim.add_soft(Implies(Not(src_bool), src_bool), constraint.weight)
                elif constraint.implication == "no_no":
                    optim.add_soft(Implies(Not(src_bool), Not(dest_bool)), constraint.weight)

        optim.check()
        mod = optim.model()

        new_beliefs = copy.deepcopy(beliefs)
        for prop in new_beliefs:
            sentence = prop.sentence
            prop.boolean = mod.evaluate(belief_bools[sentence])
        
        return new_beliefs

    def feedback(self, proposition: Proposition, feedback_type=FeedbackType.RELEVANT):
        """Returns feedback beliefs corresponding to the given proposition using one of multiple strategies."""
        
        if feedback_type == FeedbackType.ON_TOPIC_RANDOM:
            on_topic_beliefs = self.get_beliefs_by_subject(proposition.subject)
            return list(np.random.choice(on_topic_beliefs, min(self.num_random_on_topic_beliefs, len(on_topic_beliefs)), replace=False))

        elif feedback_type == FeedbackType.RELEVANT:
            # Sorting according to descending order
            clashing_beliefs = sorted(self.get_clashing_beliefs(proposition), key=lambda prop: -prop.weight)
            if len(clashing_beliefs) < self.num_relevant_beliefs:
                clashing_beliefs += list(self.beliefs.values())[:self.num_relevant_beliefs - len(clashing_beliefs)]
            return clashing_beliefs[:self.num_relevant_beliefs]
            

    def get_clashing_beliefs(self, proposition: Proposition):
        """Returns a list of beliefs present in the model that are inconsistent with the given proposition"""
        
        subject = proposition.subject
        clashing_beliefs = []

        visited_nodes = set()
        def dfs(predicate: Predicate, expected_truth: bool):
            visited_nodes.add(predicate.str_rep)
            sentence = predicate.substitute(subject)
            if self.beliefs[sentence].boolean != expected_truth:
                clashing_beliefs.append(self.beliefs[sentence])
            for constraint in self.links[predicate.str_rep]:
                if not constraint.dest_predicate.str_rep in visited_nodes:
                    if constraint.implication == "yes_yes" and expected_truth == True:
                        dfs(constraint.dest_predicate, True)
                    elif constraint.implication == "yes_no" and expected_truth == True:
                        dfs(constraint.dest_predicate, False)
                    elif constraint.implication == "no_yes" and expected_truth == False:
                        dfs(constraint.dest_predicate, True)
                    elif constraint.implication == "no_no" and expected_truth == False:
                        dfs(constraint.dest_predicate, False)

        return clashing_beliefs

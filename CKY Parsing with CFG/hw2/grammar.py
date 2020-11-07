"""
COMS W4705 - Natural Language Processing - Fall 20 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
# from math import fsum
from math import isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)
        if self.verify_grammar() == True:
            print("The grammar is a valid PCFG in CNF.")
        else:
            print("The grammar is not a valid PCFG in CNF!")
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for lhs in self.lhs_to_rules:
            all_rules = self.lhs_to_rules[lhs]
            count = 0
            for rule in all_rules:
                rhs = rule[1]
                if len(rhs) == 2:
                    for node in rhs:
                        if not node.isupper():
                            return False
                elif len(rhs) != 1:
                    return False
                count += rule[2]
            if not isclose(1, count):
                return False
        
        return True 


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
    #with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file)
        

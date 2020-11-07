"""
COMS W4705 - Natural Language Processing - Fall 2020  
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg
from math import log

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        table = defaultdict(list)
        for index in range(len(tokens)):
            word = tokens[index]
            if (word,) in self.grammar.rhs_to_rules:
                rules = self.grammar.rhs_to_rules[(word,)]
                for r in rules:
                    table[(index, index+1)].append(r[0])
            else:
                return False
        if len(tokens) == 1:
            if self.grammar.startsymbol in table[(0,1)]:
                return True
            else:
                return False
 
        for l in range(2, len(tokens)+1):
            for start in range(len(tokens)-l+1):
                for i in range(start+1, start+l):
                    if (start, i) in table and (i, start+l) in table:
                        for left in table[(start, i)]:
                            for right in table[(i, start+l)]:
                                if (left, right) in self.grammar.rhs_to_rules:
                                    for r in self.grammar.rhs_to_rules[(left, right)]:
                                        parent = r[0]
                                        if parent not in table[(start, start+l)]:
                                            table[(start, start+l)].append(parent)
                            
        if (0, len(tokens)) in table and self.grammar.startsymbol in table[(0, len(tokens))]:
            return True
        else:
            return False 
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict(dict)
        probs = defaultdict(dict)
        
        for index in range(len(tokens)):
            word = tokens[index]
            if (word,) in self.grammar.rhs_to_rules:
                rules = self.grammar.rhs_to_rules[(word,)]
                for r in rules:
                    parent = r[0]
                    curr = log(r[2])
                    if (index, index+1) in table and parent in table[(index, index+1)]:
                        max_prob = probs[(index, index+1)][parent]
                        if curr > max_prob:
                            table[(index, index+1)][parent] = word
                            probs[(index, index+1)][parent] = curr
                    else:
                        table[(index, index+1)][parent] = word
                        probs[(index, index+1)][parent] = curr
        
        for l in range(2, len(tokens)+1):
            for start in range(len(tokens)-l+1):
                for i in range(start+1, start+l):
                    if (start, i) in table and (i, start+l) in table:
                        for left in table[(start, i)]:
                            left_prob = probs[(start, i)][left]
                            for right in table[(i, start+l)]:
                                right_prob = probs[(i, start+l)][right]
                                if (left, right) in self.grammar.rhs_to_rules:
                                    for r in self.grammar.rhs_to_rules[(left, right)]:
                                        parent = r[0]
                                        curr = log(r[2])
                                        if (start, start+l) in table and parent in table[(start, start+l)]:
                                            max_prob = probs[(start, start+l)][parent]
                                            now = curr+left_prob+right_prob
                                            if now > max_prob:
                                                first = (left, start, i)
                                                second = (right, i, start+l)
                                                table[(start, start+l)][parent] = (first, second)
                                                probs[(start, start+l)][parent] = now
                                        else:
                                            now = curr+left_prob+right_prob
                                            first = (left, start, i)
                                            second = (right, i, start+l)
                                            table[(start, start+l)][parent] = (first, second)
                                            probs[(start, start+l)][parent] = now
                                        
        
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    
    children = chart[(i,j)][nt]
    if type(children) == str:
        return (nt, children)
    else:
        (left, right) = children
        left_node = left[0]
        right_node = right[0]
        left_i = left[1]
        left_j = left[2]
        right_i = right[1]
        right_j = right[2]
        left_result = get_tree(chart, left_i, left_j, left_node)
        right_result = get_tree(chart, right_i, right_j, right_node)
        return (nt, left_result, right_result)
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks))
        chart,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(chart)
        assert check_probs_format(probs)
        

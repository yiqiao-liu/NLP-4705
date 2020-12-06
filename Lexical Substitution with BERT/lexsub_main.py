#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers

import re
import string

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    output = []
    lexemes = wn.lemmas(lemma, pos)
    for lexeme in lexemes:
        synset = lexeme.synset()
        lemmas = synset.lemmas()
        for l in lemmas:
            word = l.name()
            if word != lemma and word not in output:
                if '_' in word:
                    word = re.sub('[_]', ' ', word)
                output.append(word)
    return output

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:

    '''
    Total = 298, attempted = 298
    precision = 0.098, recall = 0.098
    Total with mode 206 attempted 206
    precision = 0.136, recall = 0.136
    '''

    lemma = context.lemma
    pos = context.pos

    count_dic = {}
    lexemes = wn.lemmas(lemma, pos)
    for lexeme in lexemes:
        synset = lexeme.synset()
        lemmas = synset.lemmas()
        for l in lemmas:
            word = l.name()
            if word != lemma:
                if '_' in word:
                    word = re.sub('[_]', ' ', word)
                count = l.count()
                if word in count_dic:
                    count_dic[word] += count
                else:
                    count_dic[word] = count
    return max(count_dic, key=lambda x: count_dic[x])

def wn_simple_lesk_predictor(context : Context) -> str:

    '''
    Total = 298, attempted = 298
    precision = 0.102, recall = 0.102
    Total with mode 206 attempted 206
    precision = 0.136, recall = 0.136
    '''

    # extract the data needed
    lemma = context.lemma
    pos = context.pos
    left_context = context.left_context
    right_context = context.right_context
    all_context = left_context + right_context
    stop_words = stopwords.words('english')

    # get tuple_first = the overlap count between context and each synset
    # tuple_second = the frequency of the target word in each synset
    # tuple_third = the count of the most frequent word in each synset
    # sort the dictionary by the tuple and return the most frequent word of the chosen synset
    synset_count = {}
    lexemes = wn.lemmas(lemma, pos)
    for lexeme in lexemes:
        synset = lexeme.synset()
        definition = tokenize(synset.definition())
        for exp in synset.examples():
            definition += tokenize(exp)
        for hyp in synset.hypernyms():
            definition += tokenize(hyp.definition())
            for hyp_exp in hyp.examples():
                definition += tokenize(hyp_exp)
        overlap = []
        for word in all_context:
            if word in definition and word not in overlap and word not in stop_words:
                overlap.append(word)
        tuple_first = len(overlap)

        tuple_second = 0
        count_dic = {}
        for l in synset.lemmas():
            word = l.name()
            if word == lemma:
                c = l.count()
                tuple_second += c

            if word != lemma:
                if '_' in word:
                    word = re.sub('[_]', ' ', word)
                count = l.count()
                if word in count_dic:
                    count_dic[word] += count
                else:
                    count_dic[word] = count

        if count_dic == {}:
            possible_result = None
            possible_count = 0
        else:
            possible_result = max(count_dic, key=lambda x: count_dic[x])
            possible_count = count_dic[possible_result]
        tuple_third = [possible_count, possible_result]
        synset_count[synset] = (tuple_first, tuple_second, tuple_third)

        max_synset = max(synset_count, key=lambda x: synset_count[x])
        return synset_count[max_synset][2][1]

class Word2VecSubst(object):

    '''
    Total = 298, attempted = 298
    precision = 0.115, recall = 0.115
    Total with mode 206 attempted 206
    precision = 0.170, recall = 0.170
    '''
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        lemma = context.lemma
        pos = context.pos

        candidates = get_candidates(lemma, pos)
        largest = None
        count = 0
        for can in candidates:
            if can in self.model.wv:
                sim = self.model.similarity(lemma, can)
                if sim > count:
                    largest = can
                    count = sim

        return largest


class BertPredictor(object):

    '''
    Total = 298, attempted = 298
    precision = 0.123, recall = 0.123
    Total with mode 206 attempted 206
    precision = 0.184, recall = 0.184
    '''

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        lemma = context.lemma
        pos = context.pos
        left_context = context.left_context
        right_context = context.right_context
        all_context = ['[CLS]'] + left_context + ['[MASK]'] + right_context + ['[SEP]']
        mask_index = len(left_context) + 1
        candidates = get_candidates(lemma, pos)

        input_toks = self.tokenizer.convert_tokens_to_ids(all_context)
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][mask_index])[::-1]
        words = self.tokenizer.convert_ids_to_tokens(best_words, skip_special_tokens=False)

        dic = {}
        for can in candidates:
            if can in words:
                dic[can] = -words.index(can)

        return max(dic, key=lambda x: dic[x])


class MyPredictor(object):

    '''
    Total = 298, attempted = 298
    precision = 0.131, recall = 0.131
    Total
    with mode 206 attempted 206
    precision = 0.184, recall = 0.184
    '''

    def __init__(self, filename):
        self.model1 = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model2 = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context):

        lemma = context.lemma
        pos = context.pos
        left_context = context.left_context
        right_context = context.right_context
        all_context = ['[CLS]'] + left_context + ['[MASK]'] + right_context + ['[SEP]']
        mask_index = len(left_context) + 1
        candidates = get_candidates(lemma, pos)

        input_toks = self.tokenizer.convert_tokens_to_ids(all_context)
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model2.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][mask_index])[::-1]
        words = self.tokenizer.convert_ids_to_tokens(best_words, skip_special_tokens=False)

        dic = {}
        for can in candidates:
            if can in self.model1.wv:
                sim = self.model1.similarity(lemma, can)
                if pos == 'v':
                    dic[can] = sim
                else:
                    if can in words:
                        dic[can] = -words.index(can) + 10 * sim

        return max(dic, key=lambda x: dic[x])


    

if __name__=="__main__":


    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = MyPredictor(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

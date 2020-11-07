import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2020 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    
    """Process sequence: add START and STOP"""
    sequence.append("STOP")
    if n == 1:
        sequence.insert(0, "START")
    else:
        for num in range(n-1):
            sequence.insert(0, "START")
    
    """Put the sequence into tuples"""
    l = []  
    for pos in range(len(sequence)-n+1):
        if n == 1:
            l.append((sequence[pos],))
        else:
            t = (sequence[pos],)
            for i in range(pos+1, min(pos+n, len(sequence))):
                t += (sequence[i],)
            l.append(t)

    return l


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        self.totalwords = 0
        self.totalsentences = 0

        unigram = []
        bigram = []
        trigram = []
        ##Your code here
        for sentence in corpus:
            unigram += get_ngrams(sentence, 1)
            bigram += get_ngrams(sentence, 2)
            trigram += get_ngrams(sentence, 3)
            self.totalsentences += 1
        
        for x in unigram:
            if x in self.unigramcounts:
                self.unigramcounts[x] += 1
            else:
                self.unigramcounts[x] = 1
        
        for y in bigram:
            if y in self.bigramcounts:
                self.bigramcounts[y] += 1
            else:
                self.bigramcounts[y] = 1
                
        for z in trigram:
            if z in self.trigramcounts:
                self.trigramcounts[z] += 1
            else:
                self.trigramcounts[z] = 1
                
        for key in self.unigramcounts:
            self.totalwords += self.unigramcounts[key]

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """       
        if trigram not in self.trigramcounts:
            return 0.0
        
        bigram = trigram[:2]
        
        if trigram[-2] == 'START':
            denom = self.totalsentences
        else:
            denom = self.bigramcounts[bigram]
        
        prob = (self.trigramcounts[trigram])/denom
        
        return prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram[-1] == 'START':
            return 0.0
        if bigram not in self.bigramcounts:
            return 0.0
        
        unigram = bigram[:1]
        
        if bigram[-2] == 'START':
            denom = self.totalsentences
        else:
            denom = self.unigramcounts[unigram]
        
        prob = (self.bigramcounts[bigram])/denom
        
        return prob
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if unigram[0] == 'START':
            return 0.0

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        prob = (self.unigramcounts[unigram])/(self.totalwords)
        
        return prob

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        """return result"""         

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        first_slot = lambda1 * (self.raw_trigram_probability(trigram))
        second_slot = lambda2 * (self.raw_bigram_probability(trigram[1:]))
        third_slot = lambda3 * (self.raw_unigram_probability(trigram[2:]))
        
        return first_slot + second_slot + third_slot
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        all_trigram = get_ngrams(sentence, 3)
        prob = 0
        
        for trigram in all_trigram:
            prob += math.log2(self.smoothed_trigram_probability(trigram))
        
        return prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum_logprob = 0
        total_words = 0
        for sentence in corpus:
            total_words += len(sentence)
            sum_logprob += self.sentence_logprob(sentence)
            
        l = (1/total_words) * sum_logprob
        
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            total += 1
            p1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            p2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if p1 < p2:
                correct += 1
    
        for f in os.listdir(testdir2):
            total += 1
            p1 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            p2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if p1 < p2:
                correct += 1
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 
    
    #model = TrigramModel('hw1_data/brown_train.txt')

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    #dev_corpus = corpus_reader('hw1_data/brown_test.txt', model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print(pp)


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', 'hw1_data/ets_toefl_data/test_high', 'hw1_data/ets_toefl_data/test_low')
    #print(acc)


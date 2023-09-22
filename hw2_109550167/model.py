import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sympy import N
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        bigramfreq = dict()
        uuu = dict()
        
        for review in corpus_tokenize:
            previous_word = None
            for word in review:                
                if previous_word !=None:
                    bigramfreq[(previous_word, word)] = bigramfreq.get((previous_word, word), 0) + 1
                    uuu[previous_word] = uuu.get(previous_word, 0) + 1
                previous_word = word
        model = dict()
        for key in bigramfreq:
            numerator = bigramfreq[key]
            denominator = uuu[key[0]]
            prob = float(numerator)/float(denominator)
            if key[0] not in model:
                model[key[0]] = dict()
            if key[1] not in model[key[0]]:
                model[key[0]][key[1]] = prob
        return model, bigramfreq
        # end your code
    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        # begin your code (Part 2)
        entropy = 0.0
        length = 0
        for sentence in corpus:
            previous_word = None
            length += len(sentence)
            for word in sentence:
                if previous_word is not None:
                    if previous_word in self.model:
                        if word in self.model[previous_word]:
                            entropy+=math.log(self.model[previous_word][word], 2)
                previous_word = word
        perplexity = math.pow(2, -entropy/length)
        # end your code

        return perplexity

    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 500
        ssfeature = sorted(self.features.items(), key=lambda x:x[1], reverse=True)
        feature=[]
        corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['review']]
        for i in range(feature_num):
            feature.append(ssfeature[i][0])
        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        train_corpus_embedding = []
        
        for sentence in corpus:
            temp=[]
            for i in range(len(feature)):
                previous_word=None
                sum=0
                for word in sentence:
                    if previous_word is not None:
                        if previous_word == feature[i][0] and word == feature[i][1]:
                            sum+=1
                    previous_word = word
                temp.append(sum)
            train_corpus_embedding.append(temp)

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        test_corpus_embedding = []
        for sentence in corpus:
            temp = []
            for i in range(len(feature)):
                previous_word=None
                sum=0
                for word in sentence:
                    if previous_word is not None:
                        if previous_word == feature[i][0] and word == feature[i][1]:
                            sum+=1
                    previous_word = word
                temp.append(sum)
            test_corpus_embedding.append(temp)
        # end your code

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['[CLS]'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))

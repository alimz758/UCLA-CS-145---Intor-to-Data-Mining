from numpy import zeros, int8, log
from pylab import random
import sys
#import jieba
import nltk
from nltk.tokenize import word_tokenize 
import re
import time
import codecs
# N is # of of document
# K is # of topic
# M is # of word
# beta is probablity of word given a topic
# theta is probablity of a topic given a document
# document- word matrix, N x M : word count in a document
class PLSA(object):
    def initialize(self, N, K, M, word2id, id2word, X):
        self.word2id, self.id2word, self.X = word2id, id2word, X
        self.N, self.K, self.M = N, K, M
        # theta[i, j] : p(zj|di): 2-D matrix
        self.theta = random([N, K])
        # beta[i, j] : p(wj|zi): 2-D matrix
        self.beta = random([K, M])
        # p[i, j, k] : p(zk|di,wj): 3-D tensor
        self.p = zeros([N, M, K])
        for i in range(0, N):
            normalization = sum(self.theta[i, :])
            for j in range(0, K):
                self.theta[i, j] /= normalization;

        for i in range(0, K):
            normalization = sum(self.beta[i, :])
            for j in range(0, M):
                self.beta[i, j] /= normalization;


    def EStep(self):
        for i in range(0, self.N):
            for j in range(0, self.M):
                ## ================== YOUR CODE HERE ==========================
                ###  for each word in each document, calculate its
                ###  conditional probability belonging to each topic (update p)
                denominator = 0
                for k in range(0, self.K):
                    self.p[i, j, k] = self.theta[i, k] * self.beta[k, j]
                    denominator += self.p[i, j, k]
                for k in range(0, self.K):
                    self.p[i, j, k] /= denominator
                # ============================================================

    def MStep(self):
        # update beta
        for k in range(0, self.K):
            # ================== YOUR CODE HERE ==========================
            ###  Implement M step 1: given the conditional distribution
            ###  find the parameters that can maximize the expected likelihood (update beta)
            denominator = 0
            for m in range(0, self.M):
                self.beta[k, m] = 0
                for n in range(0, self.N):
                    self.beta[k, m] += self.X[n, m] * self.p[n, m, k]
                denominator += self.beta[k, m]
            for m in range(0, self.M):
                self.beta[k, m] /= denominator
            # ============================================================
        
        # update theta
        for i in range(0, self.N):
            # ================== YOUR CODE HERE ==========================
            ###  Implement M step 2: given the conditional distribution
            ###  find the parameters that can maximize the expected likelihood (update theta)
            for k in range(0, self.K):
                self.theta[i, k] = 0
                denominator = 0
                for m in range(0, self.M):
                    self.theta[i, k] += self.X[i, m] * self.p[i, m, k]
                    denominator += self.X[i, m]
                self.theta[i, k] /= denominator
            # ============================================================


    # calculate the log likelihood
    def LogLikelihood(self):
        loglikelihood = 0
        for i in range(0, self.N):
            for j in range(0, self.M):
                # ================== YOUR CODE HERE ==========================
                ###  Calculate likelihood function
                temp = 0
                for k in range(0, self.K):
                    temp += self.theta[i, k] * self.beta[k, j]
                if temp > 0:
                    loglikelihood += self.X[i, j] * log(second_term)
                # ============================================================
        return loglikelihood

    # output the params of model and top words of topics to files
    def output(self, docTopicDist, topicWordDist, dictionary, topicWords, topicWordsNum):
        # document-topic distribution
        file = codecs.open(docTopicDist,'w','utf-8')
        for i in range(0, self.N):
            tmp = ''
            for j in range(0, self.K):
                tmp += str(self.theta[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()
        
        # topic-word distribution
        file = codecs.open(topicWordDist,'w','utf-8')
        for i in range(0, self.K):
            tmp = ''
            for j in range(0, self.M):
                tmp += str(self.beta[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()
        
        # dictionary
        file = codecs.open(dictionary,'w','utf-8')
        for i in range(0, self.M):
            file.write(self.id2word[i] + '\n')
        file.close()
        
        # top words of each topic
        file = codecs.open(topicWords,'w','utf-8')
        for i in range(0, self.K):
            topicword = []
            ids = self.beta[i, :].argsort()
            for j in ids:
                topicword.insert(0, self.id2word[j])
            tmp = ''
            for word in topicword[0:min(topicWordsNum, len(topicword))]:
                tmp += word + ' '
            file.write(tmp + '\n')
        file.close()
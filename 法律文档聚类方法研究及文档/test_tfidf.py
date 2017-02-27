#coding:utf-8
# 一个计算tf-idf的小demo
from __future__ import division
import os
import math
from gensim import corpora,models,similarities

class Document:   # 文档类  文件名 词
    def __init__(self,words,filename):
        self.filename=filename
        self.words=words

def tfidf(documents):  # 计算tf-idf
    words={}
    for document in documents:  # 计算出现word的文档数量   idf
        for word in document.words.keys():
            if word not in words:words[word]=0
            words[word]+=1
    
    for document in documents:   #计算词频  df
        count=sum([value for value in document.words.values()])
        for word in document.words.keys():
            document.words[word]=(document.words[word]/count)*math.log(len(documents)/words[word])
    return documents

def get_tfidf(documents):  # 使用gensim计算得到tfidf
    documents=[[word for word in document.split()] for document in documents]
    dictionary = corpora.Dictionary(documents)
    n_items = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in documents]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    return corpus_tfidf

documents=[]
documents.append(Document({'a':1,'b':2,'c':3},'a.txt'))
documents.append(Document({'a':2,'d':2},'b.txt'))
documents=tfidf(documents)
for d in documents:
    for key in d.words.keys():
        print key,d.words[key]

documents=['a b b c c c','a a d d']
ds=get_tfidf(documents)
for d in ds:
    print d
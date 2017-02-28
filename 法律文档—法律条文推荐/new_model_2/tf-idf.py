#coding:utf-8
# 降维
from __future__ import division
import cPickle
import math

class Document:
	def __init__(self,words):
		self.words=words

def getDFWords(documents): #词频 保留出现次数大于3的词汇
    DF={}
    i=0
    for document in documents:
        # print 'SVM df==='+str(i)
        i+=1
        #document=cp.loads(document)
        for word in document.words:
            if word not in DF:
                DF[word]=0
            DF[word]+=1
    print len(DF)        
    list_words=[item[0] for item in DF.items() if item[1] >=10 and item[1]<=len(documents)*0.9]  # 词频
    del DF
    DF=dict((word,i+1) for i,word in enumerate(list_words))
    del list_words
    #print len(DF)
    return DF

# 去低词频  得到新的  new_contents 和 新的 new_documents
def make_new_documents(lexcion,documents,contents):
	new_contents=[]
	new_documents=[]
	for i,doc in enumerate(documents):
		words=[w for w in contents[i].split() if w in lexcion]
		new_contents.append(' '.join(words))
		words_dict={}
		for w in words:
			words_dict[w]=doc.words[w]
		new_documents.append(Document(words_dict))
	return new_contents,new_documents

def make_documents(new_contents):
	documents=[]
	for content in new_contents:
		words={}
		for w in content.split():
			words[w]=words.get(w,0)+1
		documents.append(Document(words))
	return documents

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

def select_feat_by_tfidf(documents,contents):
	new_contents=[]
	for i in range(len(documents)):
		topF=[]
		for w in sorted(documents[i].words.items(),key=lambda x:x[1],reverse=True)[:1000]:
			topF.append(w[0])
		# print len(topF)
		words=[w for w in contents[i].split() if w in topF]
		new_contents.append(' '.join(words))
	return new_contents

def main():
	new_contents=cPickle.load(open('split_contents_new.pkl'))
	print len(new_contents)
	print new_contents[4656]
	documents=make_documents(new_contents) 

	# 生成词典
	lexcion=getDFWords(documents)  # 去词频后的词典

	# 去低词频
	new_contents_del3,new_documents=make_new_documents(lexcion,documents,new_contents)
	cPickle.dump(new_contents_del3,open('new_contents_del3.pkl','w'))

	print len(new_contents[4656].split())
	print len(new_contents_del3[4656].split())
	print new_contents_del3[4656]

	documents=tfidf(new_documents)   # 计算tf-idf

	# print len(documents[0].words)
	# print documents[0].words[u'原告']
	# print documents[1].words[u'原告']
	# for w in sorted(documents[100].words.items(),key=lambda x:x[1],reverse=True):
	# 	print w[0],w[1]

	new_contents_tfidf=select_feat_by_tfidf(documents,new_contents_del3)
	print len(new_contents_tfidf)
	print new_contents_tfidf[4656]
	print len(new_contents_tfidf[4656].split())



if __name__ == '__main__':
	main()
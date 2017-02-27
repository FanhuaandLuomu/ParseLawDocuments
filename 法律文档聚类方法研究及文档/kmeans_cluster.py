#coding:utf-8
# 使用sklearn包进行聚类
# author：yinhao
# 将已切分、分词后的法律文档按首部、事实、裁判依据等进行聚类
# 可先根据首部聚类、再根据事实聚类。
from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# from gensim import corpora,models,similarities
# import logging
# logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
import os
import math
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans,AgglomerativeClustering
from gensim import corpora,models,similarities
import cchardet

class Document:
	def __init__(self,words,filename,text):
		self.filename=filename
		self.words=words
		self.text=text

def get_text(filename):
	# print filename
	global source_path
	lines=[]
	with open(source_path+os.sep+filename,'r') as f:
		for line in f:
			line=line.strip().replace(u'XX','').replace('×','')
			# if line.strip()=='':
			# 	line='None'
			lines.append(line.strip())
	# print len(lines)
	return lines

def extract(filename,key_part=['# 首部']):  # 按关键部分提取文书内容  默认只使用首部
	# print filename
	lines=get_text(filename)
	words=[]
	texts=''
	for key in key_part:
		index=lines.index(key)
		# print index
		words+=lines[index+1].decode('utf-8').split()  # 所有的词特征
		# print cchardet.detect(lines[index+1])
		texts+=lines[index+1].decode('utf-8')+' '
	words_dict={}
	for w in words:
		words_dict[w]=words_dict.get(w,0)+1   # dict
	return Document(words_dict,filename,texts)
	# return words

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
    return documents,words

def get_tfidf(documents):  # 使用gensim计算得到tfidf
	documents=[[word for word in document.text.split()] for document in documents]
	dictionary = corpora.Dictionary(documents)
	n_items = len(dictionary)
	corpus = [dictionary.doc2bow(text) for text in documents]
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]

	ds = []
	for doc in corpus_tfidf:
		d = [0] * n_items
		for index, value in doc :
			d[index]  = value
		ds.append(d)
	return ds

def create_format_mat(documents,words):
	# for word in words:
	# 	print word,words[word]
	new_words={}
	for i,w in enumerate(words.keys()):
		new_words[w]=i
	# for word in new_words:
	# 	print word,new_words[word]

	# print len(new_words)
	docs=[]
	for document in documents:
		doc=[0]*len(new_words)
		for word in document.words:
			if word in new_words:
				doc[new_words[word]]=document.words[word]
		docs.append(doc)
	return docs

def clustering(docs,n_clusters):  # 聚类 n_clusters 类别数
	kmeans_model=KMeans(n_clusters=n_clusters,random_state=1).fit(docs)  # kmeans聚类
	labels=kmeans_model.labels_
	# hmodel=AgglomerativeClustering(n_clusters=n_clusters).fit(docs)   # 层次聚类
	# labels=hmodel.labels_
	score=metrics.silhouette_score(np.array(docs),labels,metric='euclidean')  #   euclidean  评判
	return labels,score

def write2file(item_parts):  # 将结果写入文件
	for i,items in enumerate(item_parts):
		s=items[1]  # 文件名
		# print cchardet.detect(s)
		# print len(items)
		f=open(u'%s.txt' %(s),'w')
		list0=[]
		for item in items[0]:
			# item=item[0]
			list0.append('%s\t%s\n' %(item.filename.decode('GB18030'),item.text))
		f.write('\n'.join(list0))
		f.close()

# 聚类的类别默认2类
def cluster_process(filenames,key_part,s,n_clusters=2):
	documents=[]
	texts=[]
	for fname in filenames:
		# key_part: # 法院 名称、# 文书 名称、# 案号、# 首部、# 事实、# 理由、
		# # 裁判 依据、# 裁判 主文、# 尾部、# 署名、# 日期
		# = = = 标题 = = =   = = = 正文 = = =   = = = 落款 = = =
		# key_part=['# 首部']
		# 提取key_part部分
		d=extract(fname,key_part=key_part)
		documents.append(d)

	# documents,words=tfidf(documents)
	# print len(documents),len(words)
	# docs=create_format_mat(documents,words)

	docs=get_tfidf(documents)  # 使用gensim得到tfidf

	# 聚类
	# labels [0,1,0,1,1,...]
	labels,score=clustering(docs,n_clusters)

	print 'key_part:','_'.join(key_part).decode('utf-8')
	item_parts=[]
	filename_parts=[]
	for i in range(n_clusters):
		# item=[filenames[j] for j in range(len(labels)) if labels[j]==i]
		item=[documents[j] for j in range(len(labels)) if labels[j]==i]
		# print cchardet.detect(s)
		# 聚类文件名
		filename_parts.append(([filenames[j] for j in range(len(labels)) if labels[j]==i],u'%s_%s_%d' %(s,'_'.join(key_part),i)))
		item_parts.append((item,u'%s_%s_%d' %(s,'_'.join(key_part),i)))
		print 'class_%d:%d' %(i,len(item))
	# 聚类置信度
	print 'score:',score
	print '-'*20
	write2file(item_parts)  # 写入文件
	return filename_parts
 
source_path='../split_seg_case2'  # 最终版的语料
# source_path='split_seg_case'  # 已切分、已分词后的语料
# source_path='../tmp'

def main():
	filenames=os.listdir(source_path)[:]
	print 'len(filenames):',len(filenames)
	# key_part: # 法院 名称、# 文书 名称、# 案号、# 首部、# 事实、# 理由、
		# # 裁判 依据、# 裁判 主文、# 尾部、# 署名、# 日期
		# = = = 标题 = = =   = = = 正文 = = =   = = = 落款 = = =
	key_part=['# 首部']   # 首次使用首部聚类
	s='root'   # 初始root
	filename_parts_shoubu=cluster_process(filenames,key_part,s)
	filename_parts_shishi=[]
	for f_parts in filename_parts_shoubu:
		key_part=['# 裁判 主文']	# 二阶分类依据
		s=f_parts[1]   # 上一次聚类 s
		print s
		f_parts=f_parts[0]
		filename_parts_shishi.extend(cluster_process(f_parts,key_part,s,n_clusters=2))
	print len(filename_parts_shishi)

# def main():
# 	filenames=os.listdir(source_path)[:]
# 	print 'len(filenames):',len(filenames)
# 	documents=[]
# 	texts=[]
# 	for fname in filenames:
# 		# key_part: # 法院 名称、# 文书 名称、# 案号、# 首部、# 事实、# 理由、
# 		# # 裁判 依据、# 裁判 主文、# 尾部、# 署名、# 日期
# 		# = = = 标题 = = =   = = = 正文 = = =   = = = 落款 = = =
# 		key_part=['# 首部']
# 		d=extract(fname,key_part=key_part)
# 		documents.append(d)
# 	#=====
# 	# documents,words=tfidf(documents)
# 	# print len(documents),len(words)
# 	# docs=create_format_mat(documents,words)
# 	#====
# 	docs=get_tfidf(documents)  # 使用gensim得到tfidf
# 	# print len(docs),len(docs[0])
# 	# print docs[0]

# 	score_list=[]
# 	best_score=0
# 	best_cluster=0
# 	best_labels=[]
# 	for i in range(2,4):
# 		labels,score=clustering(docs,i)
# 		score_list.append(score)
# 		if score>best_score:
# 			best_score=score
# 			best_cluster=i
# 			best_labels=labels
# 		print i,score
# 		for j in range(i):
# 			print j,list(labels).count(j)
# 	print best_cluster,best_score

# 	f0=open('tmp0.txt','w')
# 	f1=open('tmp1.txt','w')
# 	list0=[]
# 	list1=[]
# 	for i,item in enumerate(best_labels):
# 		if item==0:
# 			# print cchardet.detect(filenames[i])
# 			# f0.write('%d\t%s\t%s\n' %(i+1,filenames[i].decode('GB18030'),documents[i].text))
# 			list0.append('%d\t%s\t%s\n' %(i+1,filenames[i].decode('GB18030'),documents[i].text))
# 		else:
# 			# f1.write('%d\t%s\t%s\n' %(i+1,filenames[i].decode('GB18030'),documents[i].text))
# 			list1.append('%d\t%s\t%s\n' %(i+1,filenames[i].decode('GB18030'),documents[i].text))
# 			# print documents[i].text
# 	f0.write('\n'.join(list0))
# 	f1.write('\n'.join(list1))
# 	f0.close()
# 	f1.close()

if __name__ == '__main__':
	main()
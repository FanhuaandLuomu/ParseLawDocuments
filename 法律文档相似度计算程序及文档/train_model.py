#coding:utf-8
# 训练相应的模型
from __future__ import division
import os
import sys
import math
import time
import cPickle

class Document:
	def __init__(self,words,filename):
		self.filename=filename
		self.words=words

def get_text(f_path,filename):
	# global source_path
	lines=[]
	with open(f_path+os.sep+filename,'r') as f:
		for line in f:
			line=line.strip()
			if line.strip()=='':
				line='None'
			lines.append(line)
	# print len(lines)
	return lines

def extract(lines,filename,key_part=['# 首部']):  # 按关键部分提取文书内容  默认只使用首部
	# print filename
	words=[]
	for key in key_part:
		try:
			index=lines.index(key)
			# print index
			words+=lines[index+1].decode('utf-8').split()  # 所有的词特征
		except Exception,e:
			print e
	words_dict={}
	for w in words:
		words_dict[w]=words_dict.get(w,0)+1   # dict
	# return words
	return Document(words_dict,filename)

def make_documents(f_path,filenames,key_part):  # 将文件生成所需格式
	documents=[]   # 存放所有的内容
	for filename in filenames:
		lines=get_text(f_path,filename)
		# key_part: # 法院 名称、# 文书 名称、# 案号、# 首部、# 事实、# 理由、
		# # 裁判 依据、# 裁判 主文、# 尾部、# 署名、# 日期
		# = = = 标题 = = =   = = = 正文 = = =   = = = 落款 = = =
		# key_part=['# 首部']  # 关键字联合匹配
		content=extract(lines,filename,key_part=key_part)  # 提取出所需的部分 Document实例
		if content!='':
			documents.append(content)
	return documents

def tfidf(documents):  # 计算tf-idf
	words={}
	for document in documents:  # 计算出现word的文档数量 
		for word in document.words.keys():
			if word not in words:words[word]=0
			words[word]+=1

	idf_words={}
	for document in documents:   #计算词频  df
		count=sum([value for value in document.words.values()])  # document文档中词的总数
		for word in document.words.keys():
			tf=document.words[word]/count   # tf
			idf=math.log(len(documents)/words[word])   # idf
			if word not in idf_words:
				idf_words[word]=idf    # 保存 idf
			document.words[word]=tf*idf
	print 'len(idf_words):',len(idf_words)
	return documents,idf_words

def start(train_path,key_part):
	train_filenames=os.listdir(train_path)   # 训练样本 相似案件的来源
	print 'len(train_filenames):',len(train_filenames)

	train_documents=make_documents(train_path,train_filenames,key_part)
	print 'len(train_documents):',len(train_documents)  # 语料库的大小
	train_documents,idf_words=tfidf(train_documents)
	# cPickle.dump(train_documents,open('train_documents.pkl','wb'))  # 保存 train_documents  供下次计算直接调用
	# cPickle.dump(idf_words,open('idf_words.pkl','wb'))   # 保存idf_words
	cPickle.dump([train_documents,idf_words],open('model.pkl','wb'))

def main():
	train_path=sys.argv[1]
	t0=time.time()
	key_part_dict={0:'# 法院 名称',1:'# 文书 名称',2:'# 案号',
				3:'# 首部',4:'# 事实',5:'# 理由',6:'# 裁判 依据',7:'# 裁判 主文',
				8:'# 尾部',9:'# 署名',10:'# 日期'}
	# 列表元素可以修改 0—10 分别对应上述字典中的部分 选择多个部分则可以根据多个部分计算相似度
	key_list=[3]   # 联合多部分计算 eg. [9,10]

	key_part=[]
	for k in key_list:
		key_part.append(key_part_dict[k])

	start(train_path,key_part)
	print 'cost time:%.3fs' %(time.time()-t0)

if __name__ == '__main__':
	main()
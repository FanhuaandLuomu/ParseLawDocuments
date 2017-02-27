#coding:utf-8
# 按关键词提取所需部分，并实现计算与已有文件的相似度 找出最相似的k个
# 关键部分联合查询 多样本查询
# 将得到的相似文件名保存至文件sim_file.txt
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

def tfidf(documents):  # 计算tf-idf
	words={}
	for document in documents:  # 计算出现word的文档数量 
		for word in document.words.keys():  # keys 对于每一个document唯一
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

def test_tfidf(documents,idf_words):  # 计算测试样本的tfidf
	for document in documents:
		count=sum([value for value in document.words.values()])  # document中单词个数
		for word in document.words.keys():
			if word in idf_words:
				tf=document.words[word]/count
				idf=idf_words[word]
				document.words[word]=tf*idf
	return documents

def cosine(source,target):   # 计算两文本cos距离
    numerator=sum([source[word]*target[word] for word in source if word in target])
    sourceLen=math.sqrt(sum([value*value for value in source.values()]))
    targetLen=math.sqrt(sum([value*value for value in target.values()]))
    denominator=sourceLen*targetLen
    if denominator==0:
        return 0
    else:
        return numerator/denominator

def similar(source,target):  # 计算两个文档的相似度
    return cosine(source.words,target.words)

def find_max_sim(test,targets,top_k):  # 寻找和某个文件最相似的k个文件
	similar_dict={}
	for i,doc in enumerate(targets):
		similar_dict[i]=similar(test,doc)
	# print len(similar_dict)
	similar_list=sorted(similar_dict.iteritems(),key=lambda x:x[1],reverse=True)  # 按相似度降序排序
	top_k_list=similar_list[:top_k]    # [(index,sim),(),...]
	return top_k_list

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

def process(test,target,top_k):  # 在target中寻找与test最相似的top_k个文件
	# print 'test_file:',test.filename
	top_k_list=find_max_sim(test,target,top_k)  #在target中寻找最相似的k个文件
	sim_filenames=[]
	for i,k in enumerate(top_k_list):
		fname=target[k[0]].filename
		# print i+1,fname,k[1]
		sim_filenames.append([fname,k[1]])   # filename sims
	return sim_filenames

def write2file(test_file,sim_filenames,saved_file):
	f=open(saved_file,'a')
	f.write('test_file:%s\n' %(test_file))
	for i,fname in enumerate(sim_filenames):
		f.write('top_%d:%s\t%f\n' %(i+1,fname[0],fname[1]))
	f.write('--------------------\n')
	f.close()

def start(train_path,test_path,top_k,key_part,use_model):
	test_filenames=os.listdir(test_path)  # 测试样本 需找出相似样本的案件
	print 'len(test_filenames):',len(test_filenames)
	train_filenames=os.listdir(train_path)   # 训练样本 相似案件的来源
	print 'len(train_filenames):',len(train_filenames)

	#==在线生成训练模型
	if not use_model:
		train_documents=make_documents(train_path,train_filenames,key_part)
		print 'len(train_documents):',len(train_documents)  # 语料库的大小
		train_documents,idf_words=tfidf(train_documents)

	#==保存模型
	# cPickle.dump([train_documents,idf_words],open('model.pkl','wb'))

	#==导入已训练好的模型
	else:
		model=cPickle.load(open('model.pkl','rb'))
		train_documents,idf_words=model[0],model[1]

	test_documents=make_documents(test_path,test_filenames,key_part)
	print 'len(test_documents)',len(test_documents)
	test_documents=test_tfidf(test_documents,idf_words)

	# test_documents=documents[:len(test_documents)]  # 测试样本的tf-idf
	# train_documents=documents[len(test_documents):]  # 训练样本的tf-idf

	for i,titem in enumerate(test_documents):
		print 'test_file_%d:' %(i+1),titem.filename
		# 得到和titem最相似的top_k个文档的filename和相似度 可保存至文件
		sim_filenames=process(titem,train_documents,top_k)  # [[filename,sims],...]
		# 打印相关信息
		for i,fitem in enumerate(sim_filenames):
			print 'top_%d:'%(i+1),fitem[0],fitem[1]
		print '-'*20
		write2file(titem.filename,sim_filenames,'sim_file.txt')

def main():  
	'''
	# test_path:需要测试的样本的文件夹名   
	# train_path:相似文件的来源文件夹名
	# top_k:得到最相似的top_k个案件
	'''
	test_path,train_path,top_k=sys.argv[1],sys.argv[2],sys.argv[3]  # 解析参数 
	# key_part: # 法院 名称、# 文书 名称、# 案号、# 首部、# 事实、# 理由、
	# # 裁判 依据、# 裁判 主文、# 尾部、# 署名、# 日期
	# = = = 标题 = = =   = = = 正文 = = =   = = = 落款 = = =
	t0=time.time()
	key_part_dict={0:'# 法院 名称',1:'# 文书 名称',2:'# 案号',
				3:'# 首部',4:'# 事实',5:'# 理由',6:'# 裁判 依据',7:'# 裁判 主文',
				8:'# 尾部',9:'# 署名',10:'# 日期'}
	# 列表元素可以修改 0—10 分别对应上述字典中的部分 选择多个部分则可以根据多个部分计算相似度
	key_list=[3]   # 联合多部分计算 eg. [9,10]

	key_part=[]
	for k in key_list:
		key_part.append(key_part_dict[k])

	# 选择是否加载已有的模型  True:加载   False:不加载 在线训练
	# 必须加载和计算相似度相匹配的模型 例如：key_list都=[3]   
	# 模型可在train_model.py中训练并保存
	# 由于可以选择多部分联合查询，查询方式较多，不建议使用事先训练好的模型
	use_model = False
	start(train_path,test_path,int(top_k),key_part,use_model)
	t1=time.time()
	print 'cost time:%.5f' %(t1-t0)

if __name__ == '__main__':
	main()
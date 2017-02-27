#coding:utf-8
from __future__ import division
import os
import math

class Document:   # 文档类  文件名 词
	def __init__(self,words,filename):
		self.filename=filename
		self.words=words

def get_text(filename):
	global source_path
	lines=[]
	with open(source_path+os.sep+filename,'r') as f:
		for line in f:
			line=line.strip()
			# if line.strip()=='':
			# 	line='None'
			lines.append(line)
	# print len(lines)
	return lines

def extract(lines,filename,key_part=['# 首部']):  # 按关键部分提取文书内容  默认只使用首部
	# print filename
	words=[]
	for key in key_part:
		index=lines.index(key)
	# print index
		words+=lines[index+1].decode('utf-8').split()  # 所有的词特征
	words_dict={}
	for w in words:
		words_dict[w]=words_dict.get(w,0)+1   # dict
	# return words
	return Document(words_dict,filename)

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

def cosine(source,target):
    numerator=sum([source[word]*target[word] for word in source if word in target])
    sourceLen=math.sqrt(sum([value*value for value in source.values()]))
    targetLen=math.sqrt(sum([value*value for value in target.values()]))
    denominator=sourceLen*targetLen
    if denominator==0:
        return 0
    else:
        return numerator/denominator

def similar(source,target):
    return cosine(source.centre,target.centre)

def init(documents):
    items=[]
    for document in documents:
        items.append(Item([document]))
    
    similars=[]
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            similars.append((similar(items[i],items[j]),items[i],items[j]))
    print 'len(similars):',len(similars)
    return similars

def getDocuments(item):
    documents=[]
    stack=[item]
    while len(stack)>0:
        node=stack.pop()  # 出栈
        if node==')':
            pass
        #    print ')'
        else:
            # print 'len(node.children):',len(node.children)
            if len(node.children)>1: # 孩子节点多于1个
                #print '('
                stack.append(')')  # 入栈
                for child in node.children:
                    stack.append(child)
            else:
                documents.append(node.children[0])
                #print 'o'
    return documents

def clustering(documents,k=5):   # 层次聚类
    #tfidf(documents)
    similars=init(documents)  # 初始化similars 存放所有文档的两两相似度
    while len(similars)>0:  # 原来为1
        print len(similars) 
        maxSimilar=max(similars)  # 找出具有最大相似度的两个文档
        #print maxSimilar[0]
        bestI,bestJ=maxSimilar[1].id,maxSimilar[2].id  # 最相似的两个文档的id
        # print bestI,bestJ
        newItem=Item([maxSimilar[1],maxSimilar[2]])
        # print 'len(newItem.centre):',len(newItem.centre)
        i=0
        items={}
        while i<len(similars):
            # 将与bestI和bestJ无关的保存到items
            if similars[i][1].id!=bestI and similars[i][1].id!=bestJ and similars[i][1].id not in items:
                items[similars[i][1].id]=similars[i][1]
            if similars[i][2].id!=bestI and similars[i][2].id!=bestJ and similars[i][2].id not in items:
                items[similars[i][2].id]=similars[i][2]
            # similars中和bestI、bestJ有关的元素去掉
            if similars[i][1].id==bestI or similars[i][2].id==bestI or similars[i][1].id==bestJ or similars[i][2].id==bestJ:
                del similars[i]
            else:
                i+=1
        # 计算items剩余元素与新元素的相似度
        for item in items.values():
            similars.append((similar(item,newItem),item,newItem))
        #print len(similars)
        
    print 'len(documents):',len(documents)
    print 'len(getDocuments(newItem)):',len(getDocuments(newItem))
    # for item in getDocuments(newItem):
    #     print item.filename
    print len(getDocuments(newItem.children[0])),len(getDocuments(newItem.children[1]))
    print len(getDocuments(newItem.children[1].children[0])),len(getDocuments(newItem.children[1].children[1]))
    return newItem

def make_documents(filenames):  # 将文件生成所需格式
	documents=[]   # 存放所有的内容
	for filename in filenames:
		lines=get_text(filename)
		# key_part: # 法院 名称、# 文书 名称、# 案号、# 首部、# 事实、# 理由、
		# # 裁判 依据、# 裁判 主文、# 尾部、# 署名、# 日期
		# = = = 标题 = = =   = = = 正文 = = =   = = = 落款 = = =
		key_part=['# 首部']  # 关键字联合匹配
		content=extract(lines,filename,key_part=key_part)  # 提取出所需的部分 Document实例
		if content!='':
			documents.append(content)
	# print documents[0].words,len(documents[0].words)
	documents=tfidf(documents)  # 将词频转换为tf-idf
	# print documents[0].words,len(documents[0].words)
	return documents

id=0
class Item:
    def __init__(self,children):
        self.children=children  # 孩子节点 [document1,...]
        # build centre
        self.centre={}  # 存放特征
        if len(self.children)<2:  # 孩子节点数小于2
            self.centre=self.children[0].words
        else: # 孩子节点数大于等于2
            stack=[child for child in children]  # 将其所有的孩子节点入栈
            centre={}
            count=0
            while len(stack)>0:  # 遍历栈
                node=stack.pop()  # 出栈
                if len(node.children)>1:  #  如果孩子不唯一
                    for child in node.children:  # 将所有的孩子节点入栈
                        stack.append(child)
                else:  # 孩子唯一
                    for word in node.centre:
                        if word not in centre:centre[word]=0
                        centre[word]+=node.centre[word]
                    count+=1 # 记录子树中节点的个数
            self.centre=dict([(item[0],item[1]/count) for item in centre.items()])  # 特征(tfidf)/count 取平均
#            centre={} 
#            for child in children:
#                for word in child.centre:
#                    if word not in centre:centre[word]=0
#                    centre[word]+=child.centre[word]
#            self.centre=dict([(item[0],item[1]/len(children)) for item in centre.items()])
        
        # get id
        global id
        self.id=id
        id+=1

source_path='../split_seg_case'
def main():
	filenames=os.listdir(source_path)[:]
	# for f in filenames:
	# 	print f
	print 'len(filenames):',len(filenames)
	documents=make_documents(filenames)
	print 'len(documents):',len(documents)  # 语料库的大小
	newItem=clustering(documents,k=2)
	# docs=getDocuments(newItem)
	# print type(docs[0])
	# print len(docs)
	# for item in docs:
	# 	print item.filename



if __name__ == '__main__':
	main()
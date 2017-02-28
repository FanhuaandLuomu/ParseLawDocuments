#coding:utf-8
# 基于外部训练好的embedding的lstm分类器
from __future__ import division
from lstm_process import lstm_run
import gensim
import random
import time
import cPickle
import os
ISOTIMEFORMAT='%Y-%m-%d %X'

class Document:
	def __init__(self,polarity,words):
		self.polarity=polarity
		self.words=words

# 根据案件id得到该案件涉及的法律条文  list  /暂时不用
def get_each_case_laws(sid,new_all_pieces_count_dict):
	pieces=[]
	for item in new_all_pieces_count_dict.items():
		if sid in item[1][1]:
			pieces.append(item[0])
	return pieces  
# useage:
# for item in get_each_case_laws(0,new_all_pieces_count_dict):
# 	print item

def make_documents(icontent,polarity):  # 生成documents  最大熵分类器的套路
	documents=[]
	for item in icontent:
		words={}
		for word in item.strip().split():  # 已分词 按空格split一下即可得到每个案件的词特征
			if word not in words:    
				words[word]=1    # 词特征
		documents.append(Document(polarity,words))
	return documents

# 生成案件的document,给出polarity列表
def make_document_per_case(icontent,polarity_list):
	documents=[]
	for i in range(len(icontent)):
		item=icontent[i]
		polarity=polarity_list[i]
		words={}
		for word in item.strip().split():
			if word not in words:
				words[word]=1
		documents.append(Document(polarity,words))
	return documents

# 给出polarity和对应的列表，生成documents 
def get_contents_documents(id_list,polarity,contents):  # 根据idlist从contents得到文本特征,最终生成词特征表示
	icontent=[]
	for sid in id_list: 
		icontent.append(contents[sid])  # id_list所对应的案件的 首部+事实
	documents=make_documents(icontent,polarity)
	return documents

# 未直接给出polarity，list中的案件也不是属于同一类的，生成documents的过程中动态赋予polarity
# 这样保证了tests按原来的顺序排序
def get_contents_documents_for_test(test_id_list,pos_id_list,neg_id_list,contents):
	polarity_list=[]
	icontent=[]
	for sid in test_id_list:
		polarity_list.append(1 if sid in pos_id_list else 0)
		icontent.append(contents[sid])
	documents=make_document_per_case(icontent,polarity_list)
	lenP=sum(polarity_list)  #test中正类个数
	lenN=len(polarity_list)-lenP   # test中负类个数
	return documents,lenP,lenN

# 根据law做分类器，前8k个训练 后2000个测试 得到分类性能
def split_case_by_law_1(new_all_pieces_count_dict,law):
	all_id_list=range(10132)  # 全部的案件号
	train_id_list=all_id_list[:7200]   # 前8000个训练
	val_id_list=all_id_list[7200:8000]  # 验证集
	test_id_list=all_id_list[8000:10000]  # 2000个做测试

	# pos 涉及law的案件  neg 不涉及law的案件
	pos_id_list=new_all_pieces_count_dict[law][1]  # 涉及法律law的案件号
	neg_id_list=list(set(all_id_list)^set(pos_id_list))   # 未涉及法律law的案件号  差集
	print 'pos:',len(pos_id_list),'neg:',len(neg_id_list),'all:',len(pos_id_list)+len(neg_id_list)

	# train-->(pos_train,neg_train)
	pos_train_id_list=[sid for sid in train_id_list if sid in pos_id_list]
	neg_train_id_list=[sid for sid in train_id_list if sid in neg_id_list]
	print 'pos_train:',len(pos_train_id_list),'neg_train:',len(neg_train_id_list),\
			'all:',len(pos_train_id_list)+len(neg_train_id_list)
	
	# val-->(pos_val,neg_val)
	pos_val_id_list=[sid for sid in val_id_list if sid in pos_id_list]
	neg_val_id_list=[sid for sid in val_id_list if sid in neg_id_list]
	print 'pos_val:',len(pos_val_id_list),'neg_val:',len(neg_val_id_list),\
			'all:',len(pos_val_id_list)+len(neg_val_id_list)

	# test-->[pos_test,neg_test]
	pos_test_id_list=[sid for sid in test_id_list if sid in pos_id_list]
	neg_test_id_list=[sid for sid in test_id_list if sid in neg_id_list]
	print 'pos_test:',len(pos_test_id_list),'neg_test:',len(neg_test_id_list),\
			'all:',len(pos_test_id_list)+len(neg_test_id_list)
	return pos_train_id_list,neg_train_id_list,pos_test_id_list,neg_test_id_list

def split_case_by_law_2(new_all_pieces_count_dict,law):
	#=========该段代码同split_case_by_law_1======================
	all_id_list=range(10132)
	train_id_list=all_id_list[:8000]   # 前8000个训练
	#val_id_list=all_id_list[7200:8000]
	test_id_list=all_id_list[8000:10000]  # 2000个做测试

	# pos 涉及law的案件  neg 不涉及law的案件
	pos_id_list=new_all_pieces_count_dict[law][1]  # 涉及法律law的案件号  则出现在pos_id_list中的都是+  1
	neg_id_list=list(set(all_id_list)^set(pos_id_list))   # 未涉及法律law的案件号  - 0 差集
	print 'pos:',len(pos_id_list),'neg:',len(neg_id_list),'all:',len(pos_id_list)+len(neg_id_list)

	# train-->(pos_train,neg_train)
	pos_train_id_list=[sid for sid in train_id_list if sid in pos_id_list]
	neg_train_id_list=[sid for sid in train_id_list if sid in neg_id_list]
	print 'pos_train:',len(pos_train_id_list),'neg_train:',len(neg_train_id_list),\
			'all:',len(pos_train_id_list)+len(neg_train_id_list)

	# val-->(pos_val,neg_val)
	#pos_val_id_list=[sid for sid in val_id_list if sid in pos_id_list]
	#neg_val_id_list=[sid for sid in val_id_list if sid in neg_id_list]
	random.seed(1111)
	random.shuffle(pos_train_id_list)
	random.seed(1111)
	random.shuffle(neg_train_id_list)
	pos_train_id_list,pos_val_id_list=pos_train_id_list[:-100],pos_train_id_list[-100:]
	neg_train_id_list,neg_val_id_list=neg_train_id_list[:-100],neg_train_id_list[-100:]
	print 'pos_val:',len(pos_val_id_list),'neg_val:',len(neg_val_id_list),\
			'all:',len(pos_val_id_list)+len(neg_val_id_list)
	#============================================================
	# 然而tests不分为pos和neg，直接在生成documents的时候传入不同的polarity即可
	return pos_train_id_list,neg_train_id_list,pos_val_id_list,neg_val_id_list,\
			test_id_list,pos_id_list,neg_id_list

# tests按polarity排列  可再打乱 但原始在案件中的顺序已被打乱  但方便平衡样本
def get_trains_tests_order_by_polarity(law,new_all_pieces_count_dict,contents):
	pos_train_id_list,neg_train_id_list,pos_test_id_list,neg_test_id_list=\
						split_case_by_law_1(new_all_pieces_count_dict,law)
	# 平衡训练样本  取pos_train和neg_train中较少者
	mink=min(len(pos_train_id_list),len(neg_train_id_list)) 
	pos_train_id_list,neg_train_id_list=pos_train_id_list[:mink],neg_train_id_list[:mink]

	# 平衡测试样本  /*注释则不平衡...*/
	mink2=min(len(pos_test_id_list),len(neg_test_id_list))
	pos_test_id_list,neg_test_id_list=pos_test_id_list[:mink2],neg_test_id_list[:mink2]

	#???测试样本需不需要平衡化  暂时不平衡
	# pos_train_documents=get_contents_documents(pos_train_id_list,1,contents)
	# neg_train_documents=get_contents_documents(neg_train_id_list,0,contents)
	# pos_test_documents=get_contents_documents(pos_test_id_list,1,contents)
	# neg_test_documents=get_contents_documents(neg_test_id_list,0,contents)

	# 生成词特征表示的documents
	pos_train_documents,neg_train_documents,pos_test_documents,neg_test_documents=\
		map(get_contents_documents,[pos_train_id_list,neg_train_id_list,\
									pos_test_id_list,neg_test_id_list],\
									[1,0,1,0],\
									[contents,contents,contents,contents])

	print len(pos_train_documents),len(neg_train_documents)
	print len(pos_test_documents),len(neg_test_documents)
	# print pos_train_documents[0].polarity
	# for w in pos_train_documents[0].words.keys():
	# 	print w

	k=int(len(pos_train_documents)*0.1)  # 训练集取10%做验证集  val
	trains=pos_train_documents[:-k]+neg_train_documents[:-k]
	vals=pos_train_documents[-k:]+neg_train_documents[-k:]  # trains每个类别的后20%
	tests=pos_test_documents+neg_test_documents
	print 'len(trains):',len(trains)
	print 'len(vals):',len(vals)
	print 'len(tests):',len(tests)
	# random.shuffle(trains)

	lenT=len(trains)  # 所有的训练样本数量 平衡后
	lenV=len(vals)  # 验证集的数量   与trains数量的和等于maxent程序中的trains数量
	lenP=len(pos_test_documents)  # 测试样本 正类
	lenN=len(neg_test_documents)  # 测试样本 负类

	log_file='balance_lstm_prf_OBy_polarity.txt'  # 记录prf值等结果  平衡 和 不平衡
	result_path='balance_result_OBy_polarity'
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	return trains,vals,tests,lenT,lenV,lenP,lenN,result_path,log_file

# 训练集不变 而测试集的顺序严格按照在源contents中的顺序排列
# 为了方便后期 统计每个案件涉及的所有目标law_piece时使用
# 每个法律条文piece都有一个result_lawId文件对应，其中记录了每个案件是否涉及当前的法律条文lawId
def get_trains_tests_order_by_caseId(law,new_all_pieces_count_dict,contents):
	# train_list分为pos和neg  test_list则不划分
	pos_train_id_list,neg_train_id_list,pos_val_id_list,\
				neg_val_id_list,test_id_list,pos_id_list,neg_id_list=\
				split_case_by_law_2(new_all_pieces_count_dict,law)

	# 平衡训练样本  取pos_train和neg_train中较少者
	# mink=min(len(pos_train_id_list),len(neg_train_id_list)) 
	# pos_train_id_list,neg_train_id_list=pos_train_id_list[:mink],neg_train_id_list[:mink]

	# 此函数中test不划分，所以无需对test平衡

	# 训练
	pos_train_documents=get_contents_documents(pos_train_id_list,1,contents)  # pos_train
	neg_train_documents=get_contents_documents(neg_train_id_list,0,contents)  # neg_train
	print len(pos_train_documents),len(neg_train_documents)

	# 验证
	pos_val_documents=get_contents_documents(pos_val_id_list,1,contents)
	neg_val_documents=get_contents_documents(neg_val_id_list,0,contents)
	print len(pos_val_documents),len(neg_val_documents)

	# 测试
	test_documents,lenP,lenN=get_contents_documents_for_test(test_id_list,pos_id_list,neg_id_list,contents)


	# k=int(len(pos_train_documents)*0.1)  # 训练集取20%做验证集  val
	trains=pos_train_documents+neg_train_documents
	vals=pos_val_documents+neg_val_documents  # trains每个类别的后20%
	tests=test_documents   #  测试集

	print 'len(trains):',len(trains)   #trains的顺序无所谓
	print 'len(vals):',len(vals)   # vals
	print 'len(tests):',len(tests)   # tests的顺序最好确定  按案件的顺序进行
	# random.shuffle(trains)

	#保存trains和tests vals
	cPickle.dump(trains,open('trains.pkl','wb'))
	cPickle.dump(vals,open('vals.pkl','wb'))
	cPickle.dump(tests,open('tests.pkl','wb'))

	lenT=len(trains)  # 所有的训练样本数量 平衡后
	lenV=len(vals)

	log_file='imbalance_lstm_prf_OBy_caseId.txt'  # 记录prf值等结果  平衡 和 不平衡
	result_path='imbalance_result_OBy_caseId'
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	return trains,vals,tests,lenT,lenV,lenP,lenN,result_path,log_file

# 使用lstm分类器  输入trains、tests 输出性能 并产生相应的结果文件
def trainAndTestByLstm(trains,vals,tests,my_model,result_file,law_index,lenT,lenV,lenP,lenN,log_file):
	acc=lstm_run(trains,vals,tests,my_model,result_file,law_index,lenT,lenV,lenP,lenN,log_file)
	return acc

def main():
	# 导入相关模型
	# {xx法xx条:[涉及该法律的案件个数,[id1,id2,...]]}  案件id从0开始  0~10672 共计10672个案件  法律条款1033条
	new_all_pieces_count_dict=cPickle.load(open('../new_model_2/new_all_pieces_count_dict.pkl'))
	# [item1,item2,...]  item 案件的首部+事实部分 序号即为dict中的id 共有12043个案件
	contents=cPickle.load(open('../new_model_2/split_contents_new.pkl'))  # 已分词
	my_model=gensim.models.Word2Vec.load(r'Model/model.m')
	#-------------分割线---------------

	# print new_all_pieces_count_dict[u'中华人民共和国合同法-第二百零六条'][0]
	# print new_all_pieces_count_dict[u'中华人民共和国合同法-第二百零六条'][1]
	print 'len(new_all_pieces_count_dict):',len(new_all_pieces_count_dict)
	print 'len(contents):',len(contents)
	# print contents[0]

	# 涉及案件数量超过100个的共有48个法律条文...
	for i,item in enumerate(sorted(new_all_pieces_count_dict.items(),\
							key=lambda x:x[1][0],reverse=True)[:1]):
		law=item[0]   # 法律条文
		# 一、为了能够平衡化 分离pos和neg,平衡后再合并成tests  
		# trains,vals,tests,lenT,lenV,lenP,lenN,result_path,log_file=\
		# 			get_trains_tests_order_by_polarity(law,new_all_pieces_count_dict,contents)

		# 二、为了后续多标签分类问题 还设计了tests按源顺序排列的测试集
		trains,vals,tests,lenT,lenV,lenP,lenN,result_path,log_file=\
					get_trains_tests_order_by_caseId(law,new_all_pieces_count_dict,contents)

		# if lenP==0:  # 测试样本中没有涉及law的案件  直接写文件 跳过分类器...
		# 	nowTime=time.strftime(ISOTIMEFORMAT,time.localtime())
		# 	f=open(log_file,'a')
		# 	f.write('--law_index:%d--lenT:%d--lenV:%d-lenP:%d--lenN:%d--raw_rate:%.3f--time:%s-\n' \
		# 	%(i+1,lenT,lenV,lenP,lenN,0,nowTime))
		# 	f.write('-------------None----------------\n\n')
		# 	f.close()
		# 	continue  # 跳过本次循环

		# print len(trains),len(vals),len(tests)
		# print type(my_model)

		acc=trainAndTestByLstm(trains,vals,tests,my_model,\
								result_path+os.sep+'result_%d.txt' %(i+1),i+1,\
								lenT,lenV,lenP,lenN,log_file)
		print acc

if __name__ == '__main__':
	main()
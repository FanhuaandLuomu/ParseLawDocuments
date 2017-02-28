#coding:utf-8
# 最大熵分类器 可以按法律条文（出现次数排列）做分类器，前8000训练，接着2000测试
# 两种方案：
# 一、测试集分为pos和neg，再分为平衡和不平衡
# 二、测试集不划分，保持原来在contents中的顺序 8000,8001,...。
#     方便统计每个测试案件分类器预测涉及哪些条文
import cPickle
import maxent
import random
import time
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

# icontent 中的元素都属于同一polarity 
# 生成documents列表
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
# def split_case_by_law(new_all_pieces_count_dict,law):
# 	all_id_list=range(10672)  # 全部的案件号
# 	train_id_list=all_id_list[:8000]   # 前8000个训练
# 	test_id_list=all_id_list[8000:10000]  # 2000个做测试

# 	# pos 涉及law的案件  neg 不涉及law的案件
# 	pos_id_list=new_all_pieces_count_dict[law][1]  # 涉及法律law的案件号
# 	neg_id_list=list(set(all_id_list)^set(pos_id_list))   # 未涉及法律law的案件号  差集
# 	print 'pos:',len(pos_id_list),'neg:',len(neg_id_list),'all:',len(pos_id_list)+len(neg_id_list)

# 	# train-->(pos_train,neg_train)
# 	pos_train_id_list=[sid for sid in train_id_list if sid in pos_id_list]
# 	neg_train_id_list=[sid for sid in train_id_list if sid in neg_id_list]
# 	print 'pos_train:',len(pos_train_id_list),'neg_train:',len(neg_train_id_list),\
# 			'all:',len(pos_train_id_list)+len(neg_train_id_list)

# 	# test-->[pos_test,neg_test]
# 	pos_test_id_list=[sid for sid in test_id_list if sid in pos_id_list]
# 	neg_test_id_list=[sid for sid in test_id_list if sid in neg_id_list]
# 	print 'pos_test:',len(pos_test_id_list),'neg_test:',len(neg_test_id_list),\
# 			'all:',len(pos_test_id_list)+len(neg_test_id_list)
# 	return pos_train_id_list,neg_train_id_list,pos_test_id_list,neg_test_id_list

# 根据law做分类器，前8k个训练 后2000个测试 得到分类性能  【保证tests的有序性】
def split_case_by_law_1(new_all_pieces_count_dict,law):
	all_id_list=range(10132)  # 全部的案件号
	train_id_list=all_id_list[:9000]   # 前8000个训练
	test_id_list=all_id_list[9000:10000]  # 2000个做测试

	# pos 涉及law的案件  neg 不涉及law的案件
	pos_id_list=new_all_pieces_count_dict[law][1]  # 涉及法律law的案件号
	neg_id_list=list(set(all_id_list)^set(pos_id_list))   # 未涉及法律law的案件号  差集
	print 'pos:',len(pos_id_list),'neg:',len(neg_id_list),'all:',len(pos_id_list)+len(neg_id_list)

	# train-->(pos_train,neg_train)
	pos_train_id_list=[sid for sid in train_id_list if sid in pos_id_list]
	neg_train_id_list=[sid for sid in train_id_list if sid in neg_id_list]
	print 'pos_train:',len(pos_train_id_list),'neg_train:',len(neg_train_id_list),\
			'all:',len(pos_train_id_list)+len(neg_train_id_list)

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
	#============================================================
	# 然而tests不分为pos和neg，直接在生成documents的时候传入不同的polarity即可
	return pos_train_id_list,neg_train_id_list,test_id_list,pos_id_list,neg_id_list

#law_index:法律条文的序号 从1开始...  lenT:训练样本个数(平衡) 
# lenP:测试样本正类（1）的个数  lenN:测试样本负类（0）的个数
def trainAndTest(trains,tests,result_file,law_index,lenTP,lenTN,lenP,lenN,log_file): 
	maxent.me_classify(trains,tests,result_file)  # 训练 + 测试
	pred_prob,pred_label,real_label=maxent.getPredProb(result_file)  # 解析结果
	# law_index 为法律条文的编号（顺序）,可在new_all_pieces_count_dict2.txt中查看具体的法律名称
	acc=maxent.createPRF_me(pred_label,real_label,law_index,lenTP,lenTN,lenP,lenN,log_file)
	print acc
	return pred_prob,pred_label,real_label,acc

# tests按polarity排列  可再打乱 但原始在案件中的顺序已被打乱  但方便平衡样本
def get_trains_tests_order_by_polarity(law,new_all_pieces_count_dict,contents):
	pos_train_id_list,neg_train_id_list,pos_test_id_list,neg_test_id_list=\
						split_case_by_law_1(new_all_pieces_count_dict,law)
	# 平衡训练样本  取pos_train和neg_train中较少者
	# mink=min(len(pos_train_id_list),len(neg_train_id_list)) 
	# pos_train_id_list,neg_train_id_list=pos_train_id_list[:mink],neg_train_id_list[:mink]

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
	trains=pos_train_documents+neg_train_documents
	tests=pos_test_documents+neg_test_documents

	print 'len(trains):',len(trains)   #trains的顺序无所谓
	print 'len(tests):',len(tests)   # tests的顺序最好确定  按案件的顺序进行
	# random.shuffle(trains)

	lenT=len(trains)  # 所有的训练样本数量 平衡后
	lenP=len(pos_test_documents)  # 测试样本 正类
	lenN=len(neg_test_documents)  # 测试样本 负类

	log_file='imbalance_maxent_prf_OBy_polarity.txt'  # 记录prf值等结果  平衡 和 不平衡
	result_path='imbalance_result_OBy_polarity'
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	return trains,tests,lenT,lenP,lenN,result_path,log_file

# 训练集不变 而测试集的顺序严格按照在源contents中的顺序排列
# 为了方便后期 统计每个案件涉及的所有目标law_piece时使用
# 每个法律条文piece都有一个result_lawId文件对应，其中记录了每个案件是否涉及当前的法律条文lawId
def get_trains_tests_order_by_caseId(law,new_all_pieces_count_dict,contents):
	# train_list分为pos和neg  test_list则不划分
	pos_train_id_list,neg_train_id_list,test_id_list,pos_id_list,neg_id_list=\
				split_case_by_law_2(new_all_pieces_count_dict,law)

	# 平衡训练样本  取pos_train和neg_train中较少者
	# mink=min(len(pos_train_id_list),len(neg_train_id_list)) 
	# pos_train_id_list,neg_train_id_list=pos_train_id_list[:mink],neg_train_id_list[:mink]

	# 此函数中test不划分，所以无需对test平衡

	pos_train_documents=get_contents_documents(pos_train_id_list,1,contents)  # pos_train
	neg_train_documents=get_contents_documents(neg_train_id_list,0,contents)  # neg_train

	print len(pos_train_documents),len(neg_train_documents)

	test_documents,lenP,lenN=get_contents_documents_for_test(test_id_list,pos_id_list,neg_id_list,contents)

	trains=pos_train_documents+neg_train_documents  # 训练集
	tests=test_documents   #  测试集

	print 'len(trains):',len(trains)   #trains的顺序无所谓
	print 'len(tests):',len(tests)   # tests的顺序最好确定  按案件的顺序进行
	# random.shuffle(trains)

	lenT=len(trains)  # 所有的训练样本数量 平衡后
	lenTP=len(pos_train_documents)
	lenTN=len(neg_train_documents)

	log_file='imbalance_maxent_prf_OBy_caseId.txt'  # 记录prf值等结果  平衡 和 不平衡
	result_path='imbalance_result_OBy_caseId'
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	return trains,tests,lenTP,lenTN,lenP,lenN,result_path,log_file

def main():
	# 导入相关模型
	# {xx法xx条:[涉及该法律的案件个数,[id1,id2,...]]}  案件id从0开始  0~10131 共计10132个案件  法律条款914条
	new_all_pieces_count_dict=cPickle.load(open('../new_model_2/new_all_pieces_count_dict.pkl'))
	# [item1,item2,...]  item 案件的首部+事实部分 序号即为dict中的id 共有12043个案件
	contents=cPickle.load(open('../new_model_2/split_contents_new.pkl'))  # 已分词
	#-------------分割线---------------

	# print new_all_pieces_count_dict[u'中华人民共和国合同法-第二百零六条'][0]
	# print new_all_pieces_count_dict[u'中华人民共和国合同法-第二百零六条'][1]
	print 'len(new_all_pieces_count_dict):',len(new_all_pieces_count_dict)
	print 'len(contents):',len(contents)    # 10179
	# print contents[-1]

	# for item in get_each_case_laws(,new_all_pieces_count_dict):
	# 	print item

	
	# record=[]  # 记录每次 测试样本(+-类)的个数 预测正确率
	# 涉及案件数量超过100个的共有48个法律条文...
	for i,item in enumerate(sorted(new_all_pieces_count_dict.items(),key=lambda x:x[1][0],reverse=True)[:50]):
		law=item[0]   # 法律条文
		# 一、为了能够平衡化 分离pos和neg,平衡后再合并成tests  
		# trains,tests,lenT,lenP,lenN,result_path,log_file=\
		# 			get_trains_tests_order_by_polarity(law,new_all_pieces_count_dict,contents)

		# 二、为了后续多标签分类问题 还设计了tests按与按顺序排列的测试集
		trains,tests,lenTP,lenTN,lenP,lenN,result_path,log_file=\
					get_trains_tests_order_by_caseId(law,new_all_pieces_count_dict,contents)

		# print 'len(trains):',len(trains)   #trains的顺序无所谓
		# print 'len(tests):',len(tests)   # tests的顺序最好确定  按案件的顺序进行
		# # random.shuffle(trains)

		# lenT=len(trains)  # 所有的训练样本数量 平衡后
		# lenP=len(pos_test_documents)  # 测试样本 正类
		# lenN=len(neg_test_documents)  # 测试样本 负类

		# log_file='imbalance_maxent_prf.txt'  # 记录prf值等结果  平衡 和 不平衡
		# result_path='imbalance_result'
		# if not os.path.exists(result_path):
		# 	os.mkdir(result_path)

		# if lenP==0:  # 测试样本中没有涉及law的案件  直接写文件 跳过分类器...
		# 	nowTime=time.strftime(ISOTIMEFORMAT,time.localtime())
		# 	f=open(log_file,'a')
		# 	f.write('--law_index:%d--lenT:%d--lenP:%d--lenN:%d--raw_rate:%.3f--time:%s-\n' \
		# 	%(i+1,lenT,lenP,lenN,0,nowTime))
		# 	f.write('-------------None----------------\n\n')
		# 	f.close()
		# 	continue  # 跳过本次循环

		# 训练+预测+性能解析...
		pred_prob,pred_label,real_label,acc=trainAndTest(\
											trains,tests,\
											result_path+os.sep+'result_%d.txt' %(i+1),i+1,\
											lenTP,lenTN,lenP,lenN,log_file)
		#record 记录每次的相关信息...
		# record.append([i+1,len(pos_train_documents),len(pos_test_documents),len(neg_test_documents),acc])
		

if __name__ == '__main__':
	main()
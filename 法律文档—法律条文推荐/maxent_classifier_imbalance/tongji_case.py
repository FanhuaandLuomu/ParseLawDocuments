#coding:utf-8
# 统计训练样本8000中每个案件涉及的法律条文个数
# 统计测试样本2000中每个案件真实标签涉及的法律条文的个数
# 统计测试样本2000中每个案件预测标签的个数
from __future__ import division
import cPickle
import os

# 根据案件id得到该案件涉及的法律条文  list  /暂时不用   案件id从0开始
def get_each_case_laws(sid,new_all_pieces_count_dict):
	pieces=[]
	for item in new_all_pieces_count_dict.items():
		if sid in item[1][1]:
			pieces.append(item[0])
	return pieces  

# 统计每个案件涉及的法律条文数量
def get_pieces_count(contents,new_all_pieces_count_dict):
	count_dict={}
	for i in range(len(contents))[:8000]:
		k=len(get_each_case_laws(i,new_all_pieces_count_dict))
		count_dict[k]=count_dict.get(k,0)+1
	return count_dict

def get_tests_predAndpred_pieces_count(all_case_laebls):
	test_real_count={}
	test_pred_count={}
	for i in range(len(all_case_laebls)):
		real_label,pred_label=all_case_laebls[i][0],all_case_laebls[i][1]
		k1,k2=sum(real_label),sum(pred_label)
		test_real_count[k1]=test_real_count.get(k1,0)+1
		test_pred_count[k2]=test_pred_count.get(k2,0)+1
	return test_real_count,test_pred_count

def tongji_count(count_dict,length):
	tmp=0
	for k in sorted(count_dict.items(),key=lambda x:x[0]):
		tmp+=k[1]
		print k[0],k[1],tmp/length

def main():
	# 导入相关模型
	# {xx法xx条:[涉及该法律的案件个数,[id1,id2,...]]}  案件id从0开始  0~10672 共计10672个案件  法律条款1033条
	new_all_pieces_count_dict=cPickle.load(open('../new_model_2/new_all_pieces_count_dict.pkl'))
	# [item1,item2,...]  item 案件的首部+事实部分 序号即为dict中的id 共有12043个案件
	contents=cPickle.load(open('../new_model_2/split_contents_new.pkl'))  # 已分词
	#-------------分割线---------------
	print len(new_all_pieces_count_dict)
	print len(contents)

	# 统计训练样本每个案件涉及的法律个数  {1:count1,2:count2,...}
	count_dict=get_pieces_count(contents,new_all_pieces_count_dict)
	tongji_count(count_dict,8000)
	print '-'*20

	# 导入 real_label_list 和 pred_label_list  [[r_label,p_label],...]
	all_case_laebls=cPickle.load(open('all_case_labels.pkl'))
	# 统计测试样本每个案件真实标签的个数、预测标签的个数
	test_real_count,test_pred_count=get_tests_predAndpred_pieces_count(all_case_laebls)
	tongji_count(test_real_count,2000)
	print '-'*20
	tongji_count(test_pred_count,2000)

if __name__ == '__main__':
	main()
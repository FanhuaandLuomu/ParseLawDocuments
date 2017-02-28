#coding:utf-8
# 从imbalance_result_OBy_caseId中的result_id.txt文件中读取每个案件的
# real_label_list 和 pred_label_list
# 并计算相应的多标签性能 
# 最终性能计算由eval_functions.py得到
from __future__ import division
import os
import numpy
import cPickle
from eval_functions import *

def readFromFile(filesname):
	real_label=[]
	pred_label=[]
	pred_prob=[]
	f=open(filesname)
	for line in f:
		pieces=line.strip().split()
		r_label=pieces[1]  # 真实标签
		p_label=pieces[2].split(':')[0]   # 预测标签
		p_prob=pieces[2].split(':')[1]  # 新添加预测的概率  供限制预测标签个数使用

		real_label.append(int(r_label))  # 真实标签  是否涉及该法律条文
		pred_label.append(int(p_label))  # 预测标签  预测是否涉及该法律条文
		pred_prob.append(float(p_prob))  # 预测概率  预测该案件涉及某法律条文的概率
 	return real_label,pred_label,pred_prob

def get_all_case_labels(source_path,k):  # k用于限制预测标签的个数 只保留预测概率最大的k个
	labelNum=50  # 推荐标签的个数
	filesnames=['result_%d.txt' %(i+1) for i in range(labelNum)]  # 结果文件列表
	print 'len(filesnames):',len(filesnames)

	all_case_laebls=[]
	for i in range(2000):  # 共2000个测试案件
		r_label=[0]*labelNum   # 真实label    50个法律条文
		p_label=[0]*labelNum   # 预测label
		p_prob=[0.0]*labelNum  # 预测概率
		all_case_laebls.append([r_label,p_label,p_prob])

	for i,fname in enumerate(filesnames): # 第i个法律条文  共50个
		# 得到一个法律条文对应的label
		# 真实哪些案件涉及  和  预测哪些案件涉及
		real_label,pred_label,pred_prob=readFromFile(source_path+os.sep+fname)
		for j in range(len(real_label)):  # 2000个案件  第j个案件
			if real_label[j]==1:
				all_case_laebls[j][0][i]=1    # 第j个案件的真实标签中有第i个法律
			if pred_label[j]==1:
				all_case_laebls[j][1][i]=1   # 第j个案件的预测标签中有第i个法律
				all_case_laebls[j][2][i]=pred_prob[j]  # 同时保存预测为该标签的概率值p

	if k>0:   # k>0时才执行预测标签限制操作
		new_all_case_labels=cut_pred_label(all_case_laebls,k,labelNum)
	else:
		new_all_case_labels=all_case_laebls
	# [[real_label_list,predict_label_list],...]   2000个测试案件  len(real_label_list)=50 类别
	return new_all_case_labels

# 根据k值，保留预测标签最大的前k个label 
def cut_pred_label(all_case_laebls,k,labelNum):
	
	new_all_case_labels=[]
	for i in range(len(all_case_laebls)):
		r_label,p_label,p_prob=all_case_laebls[i][0],\
							all_case_laebls[i][1],all_case_laebls[i][2]
		if sum(p_label)>k:  # 只有预测标签的数量多于k个时才截断
			p_label=[0]*labelNum
			p_prob=numpy.array(p_prob)
			argIndex=numpy.argsort(p_prob)   # argIndex p_prob [从小到大]排序的【下标】
			savedIndex=list(argIndex)[-k:]   # 取最大的k个元素的下标
			for i in range(len(p_label)):
				if i in savedIndex:  # p_label中下标在argIndex中的元素（label）改设为1
					p_label[i]=1
		new_all_case_labels.append([r_label,p_label])
	return new_all_case_labels

def createPRF_me(real_label,pred_label):  # 计算prf值 写入文件  rate ig 比率
	accCount=0
	# p1=p2=p3=p4=p5=p6=p7=0
	class_count=2
	p=[0]*class_count
	tp=[0]*class_count
	fp=[0]*class_count
	for i in range(len(pred_label)):
		t_label=int(real_label[i])   # 真实类别
		p_label=int(pred_label[i])   # 预测类别

		for index in range(class_count):  
			if t_label==index:  # t_label 等于当前类别index
				p[index]+=1     # index类别数+1
				if p_label==t_label:  # 预测类别==真实类别
					tp[p_label]+=1   
					accCount+=1
				else:            # 预测类别不等于真实类别
					fp[p_label]+=1
	
	acc=accCount/len(real_label)   # 正确率

	precision=[0]*class_count
	recall=[0]*class_count
	F1=[0]*class_count

	for i in range(class_count):
		if tp[i]+fp[i]!=0:
			precision[i]+=(tp[i]/(tp[i]+fp[i]))
		else:
			precision[i]=0
		if p[i]==0:
			recall[i]=0
		else:
			recall[i]+=(tp[i]/p[i])
		# dot*=recall[i]
		if precision[i]+recall[i]!=0:
			F1[i]+=((2*precision[i]*recall[i])/(precision[i]+recall[i]))
		else:
			F1[i]=0
	# print('acc:%.5f' %(acc))
	return precision,recall,F1,acc

def main():
	source_path='imbalance_result_OBy_caseId'
	k=0   # 对结果优化  统计案件出现的频度  截取预测概率最大的k个,98%的案件涉及的法律条文在10以内
	all_case_laebls=get_all_case_labels(source_path,k)

	# 保存all_case_optimal_labels.pkl,供计算eval_functions.py性能所需...  
	cPickle.dump(all_case_laebls,open('all_case_labels.pkl','wb'))

	print len(all_case_laebls),len(all_case_laebls[0]),\
			len(all_case_laebls[0][0]),len(all_case_laebls[0][1])
	# print all_case_laebls[0][0]
	# print all_case_laebls[0][1]

	# print all_case_laebls[15][0]
	# print all_case_laebls[15][1]  

	# print all_case_laebls[7][0]
	# print all_case_laebls[7][1]

	p_list=[]
	r_list=[]
	f_list=[]

	total_acc=0

	# print '-'*20
	for i in range(2000):
		if all_case_laebls[i][0]==all_case_laebls[i][1]:
			#print all_case_laebls[i][0]
			#print all_case_laebls[i][1]
			#print '--%d--' %i
			total_acc+=1
			# print i
		p,r,f,acc=createPRF_me(all_case_laebls[i][0],all_case_laebls[i][1])
		# print p,r,f
		p_list.append(p[1])
		r_list.append(r[1])
		f_list.append(f[1])

	# print p_list[0],r_list[0],f_list[0]
	# print p_list[15],r_list[15],f_list[15]

	print 'p_ave:',sum(p_list)/len(p_list)
	print 'r_ave:',sum(r_list)/len(r_list)
	print 'f_ave:',sum(f_list)/len(f_list)
	
	print 'total_acc:',total_acc/2000

	# result=get_PRF_subAcc(all_case_laebls)  # P R F  subAcc
	# print result['ave_p'],result['ave_r'],result['ave_f'],result['subAcc']

	# hloss=get_hloss(all_case_laebls)   # hloss
	# print hloss

	# Accuracy=get_Accuracy(all_case_laebls)  #  Accuracy
	# print Accuracy

if __name__ == '__main__':
	main()
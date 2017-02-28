#coding:utf-8
# 从all_case_labels.pkl和all_case_optimal_labels.pkl中
# 读取分类结果  计算相应的多标签分类评价指标...
from __future__ import division
import cPickle

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

# 计算multi-label的p、r、f、subAcc （子集精度-完全一致）
def get_PRF_subAcc(all_case_laebls):
	p_list=[]
	r_list=[]
	f_list=[]
	total_acc=0

	for i in range(len(all_case_laebls)):
		if all_case_laebls[i][0]==all_case_laebls[i][1]:
			total_acc+=1
			print all_case_laebls[i][0]
			print all_case_laebls[i][1]
			print '--%d--' %i
		p,r,f,acc=createPRF_me(all_case_laebls[i][0],all_case_laebls[i][1])
		p_list.append(p[1])
		r_list.append(r[1])
		f_list.append(f[1])
	ave_p=sum(p_list)/len(p_list)
	ave_r=sum(r_list)/len(r_list)
	ave_f=sum(f_list)/len(f_list)
	subAcc=total_acc/len(all_case_laebls)
	result={}
	result['ave_p']=ave_p
	result['ave_r']=ave_r
	result['ave_f']=ave_f
	result['subAcc']=subAcc
	return result

def get_per_hloss(list1,list2):  # 计算单个样本的汉明损失
	count=0
	for i in range(len(list1)):
		if list1[i]!=list2[i]:
			count+=1
	hloss=count/len(list1)
	return hloss

# 针对所有待测实例，衡量样本预测标签与样本实际标签之间的不一致程度，
# 即样本具有某标签但未被识别出来或者不具有某标签却被误判的可能性
def get_hloss(all_case_laebls):  # 计算测试集的hloss
	hloss_list=[]
	for i in range(len(all_case_laebls)):
		hloss_list.append(get_per_hloss(\
							all_case_laebls[i][0],all_case_laebls[i][1]))
	hloss=sum(hloss_list)/len(hloss_list)
	return hloss

# 精确率(Accuracy)统计正确预测的标签占预测标签与真实标签集合的比例，
def get_Accuracy(all_case_laebls):
	acc_list=[]
	for i in range(len(all_case_laebls)):
		list1,list2=all_case_laebls[i][0],all_case_laebls[i][1]
		jiaoji=0
		bingji=0
		for j in range(len(list1)):
			if list1[j]==1 and list2[j]==1:
				jiaoji+=1
			if list1[j]==1 or list2[j]==1:
				bingji+=1
		acc_list.append(0 if bingji==0 else jiaoji/bingji)
	acc=sum(acc_list)/len(acc_list)
	return acc

def main():
	
	# 导入 real_label_list 和 pred_label_list [[r_label,p_label],...]
	# 未优化的预测结果  all_case_labels.pkl
	# k=10优化的预测结果    all_case_optimal_labels.pkl
	all_case_laebls=cPickle.load(open('all_case_labels.pkl'))
	all_case_laebls=[[item[0][:50],item[1][:50],item[2][:50]] for item in all_case_laebls]

	# print all_case_laebls[158][0]
	# print all_case_laebls[158][1]

	result=get_PRF_subAcc(all_case_laebls)  # P R F  subAcc
	print 'ave_p:',result['ave_p']
	print 'ave_r:',result['ave_r']
	print 'ave_f:',result['ave_f']
	print 'subAcc:',result['subAcc']

	hloss=get_hloss(all_case_laebls)   # hloss
	print 'hloss:',hloss

	# print all_case_laebls[0][0]
	# print all_case_laebls[0][1]
	Accuracy=get_Accuracy(all_case_laebls)  #  Accuracy
	print 'Accuracy:',Accuracy


if __name__ == '__main__':
	main()
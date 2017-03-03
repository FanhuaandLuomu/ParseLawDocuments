#coding:utf-8
# baseline 
# 1.测试样本标签取出现频率最高的前10个
# 2.测试样本标签随机取10个
import cPickle
import random

# 取频率前10的类别作为测试集标签
def create_pred_label_top10(answers_test):
	pred_labels=[]
	for i in range(len(answers_test)):
		pred_labels.append([1 if i<10 else 0 for i in range(50)])
	return pred_labels

# 随机取10个类别作为标签
def create_pred_label_by_random(answers_test):
	pred_labels=[]
	for i in range(len(answers_test)):
		t=random.sample(range(50),10)
		pred_labels.append([1 if i in t else 0 for i in range(50)])
	return pred_labels

def create_all_case_labels(answers_test,pred_labels):
	all_case_labels=[]
	for i in range(len(answers_test)):
		all_case_labels.append([list(answers_test[i]),pred_labels[i]])
	return all_case_labels

def main():
	test_data=cPickle.load(open('test_data.pkl'))
	inputs_test,queries_test,answers_test=test_data[0],test_data[1],test_data[2]

	pred_labels=create_pred_label_top10(answers_test)

	all_case_labels=create_all_case_labels(answers_test,pred_labels)
	cPickle.dump(all_case_labels,open('all_case_labels.pkl','w'))





if __name__ == '__main__':
	main()
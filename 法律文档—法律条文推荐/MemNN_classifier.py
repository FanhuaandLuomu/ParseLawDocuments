#coding:utf-8
# 使用MemNN训练模型 在测试集上调参
from __future__ import print_function
from keras.models import Sequential,Model,load_model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout,Input,merge
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.utils.visualize_util import plot
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import cPickle
import os
import eval_functions
np.random.seed(1111)


# 根据案件id得到该案件涉及的法律条文[前50的法律条文]【的id，排序后，id从0开始...】
def get_each_case_laws(sid,new_all_pieces_count_dict):
	pieces=[]
	for i,item in enumerate(sorted(new_all_pieces_count_dict.items(),\
							key=lambda x:x[1][0],reverse=True)[:50]):
		if sid in item[1][1]:
			# pieces.append(item[0])  # 法律条文的内容
			pieces.append(i)   #  法律条文的序号
	return pieces 

# 得到stories  [[stories,question,answer],...]
def get_stories(s_index,e_index,contents,top_50_law_text,new_all_pieces_count_dict):
	documents=[]
	# 前50的法律条文的内容  合并
	stories=reduce(lambda x,y:x+y,top_50_law_text,[]) # [w0,w1,...]
	for i in range(s_index,e_index):   # contents中的起始结束序号   i为案件的序号
		question=contents[i].split()   # 案件描述信息 [w0,w1,w2,...]
		labels=get_each_case_laws(i,new_all_pieces_count_dict)  # 标签  [11,27,34]
		answers=[0]*50   # 序列化
		for k in labels:
			answers[k]=1
		documents.append([stories,question,answers])
		# print(i)
	return documents

# 生成词典
def get_vocab(train_stories,test_stories,test_stories2):
	vocab=sorted(reduce(lambda x,y:x|y,(set(story+q) for story,q,answer\
				 in train_stories+test_stories+test_stories2)))
	return vocab

def vectorize_stories(data,word_idx,story_maxlen,query_maxlen):
	X=[]
	Xq=[]
	Y=[]
	for story,query,answer in data:
		x=[word_idx[w] for w in story]   # 将story转化为矩阵
		xq=[word_idx[w] for w in query]
		y=np.array(answer)
		X.append(x)
		Xq.append(xq)
		Y.append(y)
	return (pad_sequences(X,maxlen=story_maxlen),
			pad_sequences(Xq,maxlen=query_maxlen),np.array(Y))

# End-To-End MemNet
def Mem_Model(story_maxlen,query_maxlen,vocab_size):
	input_encoder_m=Input(shape=(story_maxlen,),dtype='int32',name='input_encoder_m')

	x=Embedding(output_dim=64,input_dim=vocab_size,input_length=story_maxlen)(input_encoder_m)

	x=Dropout(0.5)(x)

	question_encoder=Input(shape=(query_maxlen,),dtype='int32',name='question_encoder')

	y=Embedding(output_dim=64,input_dim=vocab_size,input_length=query_maxlen)(question_encoder)

	y=Dropout(0.5)(y)

	z=merge([x,y],mode='dot',dot_axes=[2,2])
	# z=merge([x,y],mode='sum')

	match=Activation('softmax')(z)

	input_encoder_c=Input(shape=(story_maxlen,),dtype='int32',name='input_encoder_c')

	c=Embedding(output_dim=query_maxlen,input_dim=vocab_size,input_length=story_maxlen)(input_encoder_c)

	c=Dropout(0.5)(c)

	response=merge([match,c],mode='sum')

	w=Permute((2,1))(response)

	answer=merge([w,y],mode='concat',concat_axis=-1)

	lstm=LSTM(32)(answer)

	lstm=Dropout(0.5)(lstm)

	main_loss=Dense(50,activation='sigmoid',name='main_output')(lstm)

	model=Model(input=[input_encoder_m,question_encoder,input_encoder_c],output=main_loss)
	return model

def Mem_Model2(story_maxlen,query_maxlen,vocab_size):
	input_encoder_m = Sequential()
	input_encoder_m.add(Embedding(input_dim=vocab_size,
	                              output_dim=128,
	                              input_length=story_maxlen))
	input_encoder_m.add(Dropout(0.5))
	# output: (samples, story_maxlen, embedding_dim)
	# embed the question into a sequence of vectors
	question_encoder = Sequential()
	question_encoder.add(Embedding(input_dim=vocab_size,
	                               output_dim=128,
	                               input_length=query_maxlen))
	question_encoder.add(Dropout(0.5))
	# output: (samples, query_maxlen, embedding_dim)
	# compute a 'match' between input sequence elements (which are vectors)
	# and the question vector sequence
	match = Sequential()
	match.add(Merge([input_encoder_m, question_encoder],
	                mode='dot',
	                dot_axes=[2, 2]))
	match.add(Activation('softmax'))

	plot(match,to_file='model_1.png')

	# output: (samples, story_maxlen, query_maxlen)
	# embed the input into a single vector with size = story_maxlen:
	input_encoder_c = Sequential()
	# input_encoder_c.add(Embedding(input_dim=vocab_size,
	#                               output_dim=query_maxlen,
	#                               input_length=story_maxlen))
	input_encoder_c.add(Embedding(input_dim=vocab_size,
	                              output_dim=query_maxlen,
	                              input_length=story_maxlen))
	input_encoder_c.add(Dropout(0.5))
	# output: (samples, story_maxlen, query_maxlen)
	# sum the match vector with the input vector:
	response = Sequential()
	response.add(Merge([match, input_encoder_c], mode='sum'))
	# output: (samples, story_maxlen, query_maxlen)
	response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)

	plot(response,to_file='model_2.png')

	# concatenate the match vector with the question vector,
	# and do logistic regression on top
	answer = Sequential()
	answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))
	# the original paper uses a matrix multiplication for this reduction step.
	# we choose to use a RNN instead.
	answer.add(LSTM(64))
	# one regularization layer -- more would probably be needed.
	answer.add(Dropout(0.5))
	answer.add(Dense(50))
	# we output a probability distribution over the vocabulary
	answer.add(Activation('sigmoid'))

	return answer

# 根据预测的概率 将大于等于k的视作1
def get_labels_by_probs(prob_list,k):
	label_list=[]
	for prob in prob_list:
		label=[1 if item>=k else 0 for item in prob]
		label_list.append(label)
	return label_list

def generate_arrays(inputs_train,queries_train,answers_train):
	while 1:
		for i in inputs_train:
			yield ([inputs_train[i],queries_train[i],inputs_train[i]],answers_train[i])

# 合并real_labels和pred_labels  供计算评价指标使用
def merge_labels(real_labels,pred_labels):
	all_case_labels=[]
	for i in range(len(real_labels)):  # 1000
		all_case_labels.append([list(real_labels[i]),pred_labels[i]])
	return all_case_labels

def write2file(filename,flag,epoch,result,hloss,Accuracy):
	f=open(filename,'a')
	f.write('----flag:%s--epoch:%d------\n' %(flag,epoch))
	f.write('ave_p:%.3f\n' %(result['ave_p']))
	f.write('ave_r:%.3f\n' %(result['ave_r']))
	f.write('ave_f:%.3f\n' %(result['ave_f']))
	f.write('subAcc:%.3f\n' %(result['subAcc']))
	f.write('hloss:%.3f\n' %(hloss))
	f.write('Accuracy:%.3f' %(Accuracy))
	f.write('---------------------------\n')
	f.close()

def main():
	model_path='../new_model_2'
	# 导入相关模型
	# {xx法xx条:[涉及该法律的案件个数,[id1,id2,...]]}  案件id从0开始  0~10131 共计10132个案件  法律条款914条
	new_all_pieces_count_dict=cPickle.load(open(model_path+os.sep+'new_all_pieces_count_dict.pkl'))
	# [item1,item2,...]  item 案件的首部+事实部分 序号即为dict中的id 共有10132个案件
	# 相当于questions  10132个 列表
	contents=cPickle.load(open(model_path+os.sep+'split_contents_new.pkl'))  # 已分词
	# my_model=gensim.models.Word2Vec.load(r'Model/model.m')
	# 50条法律条文的内容
	top_50_law_text=cPickle.load(open(model_path+os.sep+'split_law_text.pkl'))
	top_50_law_text=[item.split() for item in top_50_law_text]
	#===========================
	print('len(new_all_pieces_count_dict):',len(new_all_pieces_count_dict))
	print('len(contents):',len(contents))
	print('len(law_text):',len(top_50_law_text))
	# print top_50_law_text[1]

	# 得到每个样本的标签 multi-label  从0编号
	# pieces=get_each_case_laws(0,new_all_pieces_count_dict)
	# print pieces   #  [0,2,...]

	train_stories=get_stories(280,1000,contents,top_50_law_text,new_all_pieces_count_dict)
	test_stories=get_stories(200,280,contents,top_50_law_text,new_all_pieces_count_dict)
	test_stories2=get_stories(0,200,contents,top_50_law_text,new_all_pieces_count_dict)
	# test_stories2=get_stories(8000,10000,contents,top_50_law_text,new_all_pieces_count_dict)
	print('len(train_stories):',len(train_stories))
	print('len(test_stories):',len(test_stories))
	print('len(test_stories2):'),len(test_stories2)

	# print train_stories[0][0]
	# print train_stories[0][1]
	# print train_stories[0][2]

	# pieces=get_each_case_laws(0,new_all_pieces_count_dict)
	# print pieces   #  [0,2,...]

	# 根据train_stories和test_stories生成词典
	vocab=get_vocab(train_stories,test_stories,test_stories2)
	vocab_size=len(vocab)
	print('vocab_size:',vocab_size)

	# story_maxlen=max(map(len,(x for x,_,_ in train_stories+test_stories)))
	# query_maxlen=max(map(len,(x for _,x,_ in train_stories+test_stories)))
	story_maxlen=1000
	query_maxlen=1000

	print('-')
	print('Vocab size:', vocab_size, 'unique words')
	print('Story max length:', story_maxlen, 'words')
	print('Query max length:', query_maxlen, 'words')
	print('Number of training stories:', len(train_stories))
	print('Number of test stories:', len(test_stories))
	print('Number of test stories2:', len(test_stories2))
	print('-')

	# 将词特征转化为 vector
	word_idx=dict((c,i) for i,c in enumerate(vocab))  # 字典下标从0开始

	inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
	inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)
	inputs_test2, queries_test2, answers_test2 = vectorize_stories(test_stories2, word_idx, story_maxlen, query_maxlen)

	print('-')
	print('inputs: integer tensor of shape (samples, max_length)')
	print('inputs_train shape:', inputs_train.shape)
	print('inputs_test shape:', inputs_test.shape)
	print('inputs_test2 shape:', inputs_test2.shape)
	print('-')
	print('queries: integer tensor of shape (samples, max_length)')
	print('queries_train shape:', queries_train.shape)
	print('queries_test shape:', queries_test.shape)
	print('queries_test2 shape:', queries_test2.shape)
	print('-')
	print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
	print('answers_train shape:', answers_train.shape)
	print('answers_test shape:', answers_test.shape)
	print('answers_test2 shape:', answers_test2.shape)
	print('-')
	print('Compiling...')

	model=Mem_Model2(story_maxlen,query_maxlen,vocab_size)

	# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
	model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

	plot(model,to_file='model.png')

	nb_epoch=100
	for i in range(nb_epoch):
		model.fit([inputs_train,queries_train,inputs_train],answers_train,
			batch_size=16,nb_epoch=1,verbose=1,
			validation_data=([inputs_test2, queries_test2, inputs_test2], answers_test2))

		print('epoch:%d' %(i+1))
		# 验证集
		prob_list=model.predict_proba([inputs_test,queries_test,inputs_test],batch_size=32)
		pred_labels=get_labels_by_probs(prob_list,0.5)
		all_case_labels=merge_labels(answers_test,pred_labels) 
		result=eval_functions.get_PRF_subAcc(all_case_labels)  # P R F  subAcc
		print('ave_p:',result['ave_p'])
		print('ave_r:',result['ave_r'])
		print('ave_f:',result['ave_f'])
		print('subAcc:',result['subAcc'])

		hloss=eval_functions.get_hloss(all_case_labels)   # hloss
		print('hloss:',hloss)

		# print all_case_labels[0][0]
		# print all_case_labels[0][1]
		Accuracy=eval_functions.get_Accuracy(all_case_labels)  #  Accuracy
		print('Accuracy:',Accuracy)

		write2file('result.txt','val',i+1,result,hloss,Accuracy)

		# 测试集
		prob_list=model.predict_proba([inputs_test2,queries_test2,inputs_test2],batch_size=32)
		pred_labels=get_labels_by_probs(prob_list,0.5)
		all_case_labels=merge_labels(answers_test2,pred_labels) 
		result=eval_functions.get_PRF_subAcc(all_case_labels)  # P R F  subAcc
		print('ave_p:',result['ave_p'])
		print('ave_r:',result['ave_r'])
		print('ave_f:',result['ave_f'])
		print('subAcc:',result['subAcc'])

		hloss=eval_functions.get_hloss(all_case_labels)   # hloss
		print('hloss:',hloss)

		# print all_case_labels[0][0]
		# print all_case_labels[0][1]
		Accuracy=eval_functions.get_Accuracy(all_case_labels)  #  Accuracy
		print('Accuracy:',Accuracy)

		write2file('result.txt','test',i+1,result,hloss,Accuracy)

		model.save('model/MemNN_model_epoch-%d.h5' %(i+1))  # 保存模型  包含结构、权重
		# model=load_model('model/MemNN_model_epoch-1.h5')

	'''
	nb_epoch=60
	model.fit([inputs_train, queries_train, inputs_train], answers_train,
		batch_size=16,nb_epoch=nb_epoch,verbose=1,
		validation_data=([inputs_test, queries_test, inputs_test], answers_test))

	# with open('model.json','w') as f:
	#     f.write(model.to_json())
	# model.save_weights('weights.hdf5')

	# for item in answers_test:
	# 	print(item)
	print(answers_test[7])

	#验证集
	prob_list=model.predict_proba([inputs_test,queries_test,inputs_test],batch_size=32)
	print(prob_list[7])

	# p_class=model.predict_classes([inputs_test,queries_test,inputs_test],batch_size=32)
	# print(p_class[7])

	# score=model.evaluate([inputs_test,queries_test,inputs_test],answers_test,verbose=1,
	# 													batch_size=32)
	# print(score)

	# model.save('MemNN_model_epoch-%d.hdf5' %(nb_epoch))  # 保存模型  包含结构、权重
	# cPickle.dump(model,open('MemNN_model.pkl','w'))
	# model=load_model('MemNN_model.hdf5')  #导入模型  可接着训练

	pred_labels=get_labels_by_probs(prob_list,0.5)

	# cPickle.dump(answers_test,open('real_labels.pkl','w'))
	# cPickle.dump(pred_labels,open('pred_labels.pkl','w'))

	all_case_labels=merge_labels(answers_test,pred_labels)  #  [[real_labels,pred_labels],...]
	cPickle.dump(all_case_labels,open('all_case_labels.pkl','w'))

	result=eval_functions.get_PRF_subAcc(all_case_labels)  # P R F  subAcc
	print('ave_p:',result['ave_p'])
	print('ave_r:',result['ave_r'])
	print('ave_f:',result['ave_f'])
	print('subAcc:',result['subAcc'])

	hloss=eval_functions.get_hloss(all_case_labels)   # hloss
	print('hloss:',hloss)

	# print all_case_labels[0][0]
	# print all_case_labels[0][1]
	Accuracy=eval_functions.get_Accuracy(all_case_labels)  #  Accuracy
	print('Accuracy:',Accuracy)
	
	###################################################################
	##测试
	# test_data=cPickle.load(open('test_data.pkl'))
	# inputs_test2,queries_test2,answers_test2=test_data[0],test_data[1],test_data[2]

	prob_list=model.predict_proba([inputs_test2,queries_test2,inputs_test2],batch_size=32)
	# print(prob_list[7])

	pred_labels=get_labels_by_probs(prob_list,0.5)

	# cPickle.dump(answers_test,open('real_labels.pkl','w'))
	# cPickle.dump(pred_labels,open('pred_labels.pkl','w'))

	all_case_labels=merge_labels(answers_test2,pred_labels)  #  [[real_labels,pred_labels],...]
	cPickle.dump(all_case_labels,open('all_test_case_labels.pkl','w'))

	result=eval_functions.get_PRF_subAcc(all_case_labels)  # P R F  subAcc
	print('ave_p:',result['ave_p'])
	print('ave_r:',result['ave_r'])
	print('ave_f:',result['ave_f'])
	print('subAcc:',result['subAcc'])

	hloss=eval_functions.get_hloss(all_case_labels)   # hloss
	print('hloss:',hloss)

	# print all_case_labels[0][0]
	# print all_case_labels[0][1]
	Accuracy=eval_functions.get_Accuracy(all_case_labels)  #  Accuracy
	print('Accuracy:',Accuracy)
	'''



if __name__ == '__main__':
	main()
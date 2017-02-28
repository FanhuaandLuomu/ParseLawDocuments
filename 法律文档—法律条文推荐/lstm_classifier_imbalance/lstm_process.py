#coding:utf-8
# lstm分类器主要部分 供em_lstm.py调用
from __future__ import division
import gensim
import random
import time
import os
import cPickle
ISOTIMEFORMAT='%Y-%m-%d %X'
import numpy as np  # keras每次产生确定的数据
np.random.seed(1333)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta

maxlen=1000
nb_classes=2

batch_size=32
embedding_dim=100
lstm_output_dim=128
hidden_dim=64

class Document:
	def __init__(self,polarity,words):
		self.polarity=polarity
		self.words=words

def createVec(docs,my_model):  # 生成词向量表示
	train_vec=[]
	train_label=[]
	for doc in docs:
		sen=[]
		for word in doc.words:
			if word in my_model:
				sen.append(my_model[word])  # 词向量特征
		t=len(sen)
		if t==0:  # 句子长度为0
			if len(train_vec)==0:
				s=[]
				for i in range(100):
					s.append(random.random())  # 随机生成一个词，100维随机词向量表示
				sen.append(s)
				t=1
			else:
				train_vec.append(train_vec[0])  # 用第一个样本替换
				train_label.append(doc.polarity)
				continue
		if t<maxlen:  # sen的长度t小于maxlen，padding  循环补充
			index=0
			while len(sen)<maxlen:
				sen.append(sen[index])
				index+=1
		else:
			sen=sen[:maxlen]  # 截断
		train_vec.append(sen)
		train_label.append(doc.polarity)
	return train_vec,train_label

nb_epoch=30
def LSTM_model(X_train,Y_train,X_val,Y_val,X_test,Y_test,test_label):
	print('Loading embedding successful!')
	print('len(X_train):'+str(len(X_train)))
	print('len(X_val):'+str(len(X_val)))
	print('len(X_test):'+str(len(X_test)))
	print('len(Y_train):'+str(len(Y_train)))
	print('len(Y_val):')+str(len(Y_val))
	print('len(Y_test):'+str(len(Y_test)))
	# print(test_label)
	print('X_train shape:',X_train.shape)
	print('X_val shape:',X_val.shape)
	print('X_test shape:',X_test.shape)
	print('Build model...') 

	model=Sequential()

	model.add(LSTM(lstm_output_dim,input_shape=(maxlen,embedding_dim)))  # lstm 100->128

	model.add(Dense(hidden_dim))  # 隐藏层  全连接 128->64
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	best_acc=0   # 保存验证集最好的acc
	best_pred_label=[]
	best_pred_prob=[]
	best_epoch=0
	a0=0  # 保存测试集最好的acc
	b0=0  # a0对应的epoch次数
	# best_weights_filepath='./best_weights.hdf5'
	# saveBestModel=callbacks.ModelCheckpoint(best_weights_filepath,monitor='val_loss',verbose=1,
	# 					save_best_only=True,mode='auto')
	for i in range(nb_epoch):
		print('%d epoch...' %(i+1))
		print('Now:best_acc:%.5f \t best_epoch:%d' %(best_acc,best_epoch))
		hist=model.fit(X_train,Y_train, batch_size=128, nb_epoch=1, verbose=1, shuffle=True,  #20 10
				validation_data=(X_val,Y_val))
		# print hist.history
		acc=max(hist.history['val_acc'])  # 迭代一次 max 就是唯一的那个元素val_acc
		# acc=model.evalute(X_val,Y_val,batch_size=32,show_accuracy=True,verbose=1)[0]
		p_label=model.predict_classes(X_test,batch_size=64,verbose=1)
		p_prob=model.predict_proba(X_test,batch_size=64,verbose=1)
		# print p_label
		a1=np_utils.accuracy(p_label,test_label)  # 每次迭代完在测试集上计算acc
		print 'Now epoch test acc:%.5f' %(a1)
		if a1>a0:
			print 'a1 better:%.5f' %(a1)
			a0=a1   # a0保存最好的测试集acc  更新
			b0=(i+1)   # 测试集最好acc对应的epoch
		if acc>best_acc:
			print('occured better acc,update acc and epoch...')
			best_acc=acc  # 更新val上最好的acc
			best_pred_label=p_label  # 保存val上acc最好的模型对test的预测label
			best_pred_prob=p_prob
			best_epoch=(i+1)   # val上acc最好时model对应的epoch
	test_acc=np_utils.accuracy(best_pred_label,test_label)  # 得到val上最好model在test上的正确率
	print('the best epoch:%d,and the test_acc:%.5f.,while best test acc epoch:%d,%.5f' %(best_epoch,test_acc,b0,a0))
	# print('the best pred_class:\n')
	# print(best_pred_label)
	# write2File(best_pred_label,test_label)
	return best_pred_label,best_epoch,best_pred_prob

# 使用 save_best的机制
def LSTM_model2(X_train,Y_train,X_val,Y_val,X_test,Y_test,test_label):
	print('Loading embedding successful!')
	print('len(X_train):'+str(len(X_train)))
	print('len(X_val):'+str(len(X_val)))
	print('len(X_test):'+str(len(X_test)))
	print('len(Y_train):'+str(len(Y_train)))
	print('len(Y_val):')+str(len(Y_val))
	print('len(Y_test):'+str(len(Y_test)))
	# print(test_label)
	print('X_train shape:',X_train.shape)
	print('X_val shape:',X_val.shape)
	print('X_test shape:',X_test.shape)
	print('Build model...')

	model=Sequential()

	# 栈式lstm
	# model.add(LSTM(lstm_output_dim,return_sequences=True,\
	# 				input_shape=(maxlen,embedding_dim)))
	# model.add(LSTM(lstm_output_dim,return_sequences=True))
	# model.add(LSTM(lstm_output_dim))

	model.add(LSTM(lstm_output_dim,input_shape=(maxlen,embedding_dim)))

	model.add(Dense(hidden_dim))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	# model.add(Dense(hidden_dim))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	# optmr=Adadelta(lr=0.9,rho=0.90,epsilon=1e-08)
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	plot(model,to_file='model.png')

	checkpointer=ModelCheckpoint(filepath='best_model.hdf5',monitor='val_acc',verbose=1,\
								save_best_only=True,mode='max')

	# history=LossHistory()

	hist=model.fit(X_train,Y_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,  #20 10
				validation_data=(X_val,Y_val),callbacks=[checkpointer])
	# print(history.losses)
	print hist.history

	model.load_weights('best_model.hdf5') 

	# score=model.evaluate(X_test,Y_test,batch_size=32,verbose=1)
	# print 'score:',score

	#p_label=model.predict_classes(X_test,batch_size=32,verbose=1)  # 对测试集进行预测
	p_prob=model.predict_proba(X_test,batch_size=32,verbose=1)
	p_label=np.array([np.argsort(item)[-1] for item in p_prob])
	test_acc=np_utils.accuracy(p_label,test_label)

	return p_label,p_prob

def get_documents(trains,vals,tests,my_model):  # 从em_lstm.py中接收数据供实验所用 trains,tests,model
	print 'len(trains):',len(trains)
	print 'len(vals):',len(vals)
	print 'len(tests):',len(tests)

	random.shuffle(trains)
	random.shuffle(vals)
	# random.shuffle(tests)

	train_vec,train_label=createVec(trains,my_model)  # 生成词向量表示
	val_vec,val_label=createVec(vals,my_model)
	test_vec,test_label=createVec(tests,my_model)

	X_train=np.array(train_vec)  # 转化为array矢量表示
	X_val=np.array(val_vec)
	X_test=np.array(test_vec)

	Y_train=np_utils.to_categorical(train_label,nb_classes)
	Y_val=np_utils.to_categorical(val_label,nb_classes)
	Y_test=np_utils.to_categorical(test_label,nb_classes)

	return X_train,Y_train,X_val,Y_val,X_test,Y_test,test_label

def createResult(pred_label,real_label,law_index,lenT,lenV,lenP,lenN,log_file):  # 计算prf值 写入文件
    accCount=0
    classCount=2
    p=[0]*classCount
    tp=[0]*classCount
    fp=[0]*classCount
    for i in range(len(pred_label)):
        t_label=int(real_label[i])  # 真实标签
        p_label=int(pred_label[i])  # 预测标签

        for index in range(classCount):  # 类别 0~classCount-1
            if t_label==index:  # 真是类别==index
                p[index]+=1     # 真实类别是index的个数+1
                if p_label==t_label:  # 预测为1 即预测为p_label 且正确的个数
                    tp[p_label]+=1
                    accCount+=1
                else:   # 预测为p_label 且真实为t_label 预测错误
                    fp[p_label]+=1
    acc=accCount/len(pred_label)

    precision=[0]*classCount
    recall=[0]*classCount
    F1=[0]*classCount
    # dot=1.0
    nowTime=time.strftime(ISOTIMEFORMAT,time.localtime())
    f=open(log_file,'a')   # ...prf.txt
    f.write('--law_index:%d--lenT:%d--lenV:%d--lenP:%d--lenN:%d--raw_rate:%d--time:%s--\n'\
    			%(law_index,lenT,lenV,lenP,lenN,max(lenP,lenN)/(lenP+lenN),nowTime))
    for i in range(classCount):
    	if tp[i]+fp[i]!=0:
        	precision[i]+=(tp[i]/(tp[i]+fp[i]))  # 分母为预测为i类别的个数
        else:
        	precision[i]=0

        if p[i]!=0:
        	recall[i]+=(tp[i]/p[i])   # 分母为真实为i类别的个数
        else:
        	recall[i]=0
        # dot*=recall[i]
        if precision[i]+recall[i]!=0:
        	F1[i]+=((2*precision[i]*recall[i])/(precision[i]+recall[i]))
        else:
        	F1[i]=0
        f.write('label:%d \t precision:%.5f \t recall:%.5f \t F1:%.5f\n' %(i,precision[i],recall[i],F1[i]))
    f.write('------------acc:%.5f------------\n\n' %(acc))
    f.close()
    return acc

# 将真实标签  预测标签  预测概率 写入文件result_index.txt
def make_result_file(best_pred_label,best_pred_prob,real_label,result_file):
	f=open(result_file,'w')
	for i in range(len(real_label)):
		r_label=real_label[i]  # 真实label
		p_label=best_pred_label[i]  # 预测label
		if p_label==0:
			s='%s %s %s:%s %s:%s' %(i+1,r_label,0,best_pred_prob[i][0],1,best_pred_prob[i][1])
		else:
			s='%s %s %s:%s %s:%s' %(i+1,r_label,1,best_pred_prob[i][1],0,best_pred_prob[i][0])
		f.write(s+'\n')
	f.close()

def lstm_run(trains,vals,tests,my_model,result_file,law_index,lenT,lenV,lenP,lenN,log_file):  # log_file trains tests my_model
	X_train,Y_train,X_val,Y_val,X_test,Y_test,test_label=\
									get_documents(trains,vals,tests,my_model)
	best_pred_label,best_pred_prob = LSTM_model2(\
											X_train,Y_train,X_val,Y_val,X_test,Y_test,test_label)
	acc=createResult(best_pred_label,test_label,law_index,lenT,lenV,lenP,lenN,log_file)
	make_result_file(best_pred_label,best_pred_prob,test_label,result_file)
	print acc
	return acc

# def main():
# 	t0=time.time()
# 	log_file='balance_lstm_prf.txt'
# 	trains=cPickle.load(open('trains.pkl','rb'))
# 	vals=cPickle.load(open('vals.pkl','rb'))
# 	tests=cPickle.load(open('tests.pkl','rb'))
# 	my_model=gensim.models.Word2Vec.load(r'Model/model.m')
# 	result_file='balance_result_OBy_polarity'+os.sep+'result_1.txt'
# 	law_index=1
# 	lenT,lenV,lenP,lenN=4440,1110,567,1433

# 	lstm_run(trains,vals,tests,my_model,result_file,law_index,lenT,lenV,lenP,lenN,log_file)
# 	print 'cost:%.2fs' %(time.time()-t0)

# if __name__ == '__main__':
# 	main()

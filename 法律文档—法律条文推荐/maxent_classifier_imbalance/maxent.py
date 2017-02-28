#coding:utf-8
from __future__ import division
import numpy
import math
import subprocess
import time
ISOTIMEFORMAT='%Y-%m-%d %X'

mallet_path='c:\\mallet\\dist;c:\\mallet\\dist\\mallet.jar;c:\\mallet\\dist\\mallet-deps.jar'

def get_lexcion(documents): #生成字典
	words=[]
	for document in documents:
		words+=document.words.keys()
	words=set(words) #唯一化
	lexcion=dict([(word,i+1) for i,word in enumerate(words)]) #字典 {word:id,....}
	return lexcion

def getDFWords(documents): #词频 保留出现次数大于5的词汇
    DF={}
    i=0
    for document in documents:
        # print 'SVM df==='+str(i)
        i+=1
        #document=cp.loads(document)
        for word in document.words:
            if word not in DF:
                DF[word]=0
            DF[word]+=1
    print len(DF)        
    list_words=[item[0] for item in DF.items() if item[1] >=0]  # 词频
    del DF
    DF=dict((word,i+1) for i,word in enumerate(list_words))
    del list_words
    #print len(DF)
    return DF

def createFormatText(documents,lexcion,path): #生成mallet工具所需的文本格式
	lines=[]
	for i,document in enumerate(documents):
		if(document.polarity==0):   # p 0
			line='%d 0 ' %(i+1)
		elif(document.polarity==1):
			line='%d 1 ' %(i+1)  # n 1
		elif(document.polarity==2):
			line='%d 2 ' %(i+1)  # 
		elif(document.polarity==3):
			line='%d 3 ' %(i+1)  # 
		elif(document.polarity==4):
			line='%d 4 ' %(i+1)  # 
		elif(document.polarity==5):
			line='%d 5 ' %(i+1)  # 
		elif(document.polarity==6):
			line='%d 6 ' %(i+1)  # 
		pairs=[(lexcion[word],document.words[word]) for word in document.words.keys()\
				if word in lexcion]
		line+=' '.join(['%d:%d' %(pair[0],pair[1]) for pair in pairs])
		lines.append(line)
	text='\n'.join(lines)
	f=open(path,'w')
	f.write(text)
	f.close()

def file2Bin(FilePath,BinPath): #tests.txt时要加上pipe
	if(FilePath=='test.txt'):
		s='--use-pipe-from train.bin'
	else:
		s=''
	cmd='java -cp %s lltCsv2Vectors %s --input %s --output %s'\
		%(mallet_path,s,FilePath,BinPath)
	retcode=subprocess.Popen(cmd.split())
	retcode.wait()
	if(retcode<0):
		print('To BinFile Failed！')
	else:
		print('%s to binFile success!' %FilePath)

def train(trainBinPath,modelPath):
	cmd='java -cp %s cc/mallet/classify/tui/Vectors2Classify --input %s --output-classifier \
		 %s --trainer MaxEnt' %(mallet_path,trainBinPath,modelPath)  # SVM   
	retcode=subprocess.Popen(cmd.split())
	retcode.wait()
	if(retcode<0):
		print('Train failed!')
	else:
		print('train model success!')

def classify(modelPath,testBinPath,resultPath):
	cmd='java -cp %s lltClassification --classifier %s --testing-file %s --report test:raw'\
		%(mallet_path,modelPath,testBinPath)
	retcode=subprocess.Popen(cmd.split(),stdout=open(resultPath,'w'))
	retcode.wait()
	if(retcode<0):
		print('classify failed!')
	else:
		print('classify success!')

def createResult2(resultPath):
	lines=open(resultPath,'rb').readlines()
	results=[]
	#p=n=0
	#pt=pf=0
	#nt=nf=0
	tureNum=0
	for line in lines:
		r=line.strip().split()[1:3]
		real_label=r[0]
		label,prob=r[1].split(':')
		print(real_label,label)
		if(real_label==label):
			tureNum+=1
			print 'acc:',tureNum
	acc=tureNum/len(lines)
	print('accuracy :'+str(acc))
	return acc

# def createResult(tests,resultPath):
# 	input=open(resultPath,'rb')
# 	results=[]

# 	p=n=0 #p 和n 类别样本的数量
# 	pt=0 
# 	pf=0
# 	nt=0
# 	nf=0
# 	for i,line in enumerate(input):
# 		label,prob=line.split()[2].split(':')
# 		#得到预测的类别和概率
# 		if(label==0):  #results 存放各样本属于哪个类别的可能性 -表示n类
# 			results.append(-1*float(prob))
# 		else:
# 			results.append(float(prob))

# 		if(tests[i].polarity==0): # p类
# 			p+=1   #记录p类样本的个数
# 			if(label=='0'):
# 				pt+=1  #p类预测正确的个数
# 			else:
# 				pf+=1  #p类预测错误的概率
# 		else:    #n类
# 			n+=1   #记录n类样本的个数
# 			if(label=='1'):
# 				nt+=1  #n类预测正确的个数
# 			else:
# 				nf+=1  #n类预测错误的概率

# 	acc=(pt+nt)/(p+n)
# 	print(p+n)
# 	print('accuracy:%f' %acc)

def getPredLabel(FilePath): # 得到预测的类别和真实的类别
	pred_label=[]
	real_label=[]
	f=open(FilePath,'rb')
	for line in f:
		pieces=line.strip().split()
		real_l=pieces[1]
		pred_l=pieces[2].split(':')[0]
		pred_label.append(pred_l)
		real_label.append(real_l)
	# print pred_label
	# print real_label
	return pred_label,real_label

def createPRF(FilePath,seed,log_path='maxent_prf.txt'):  # 计算prf值 写入文件
	pred_label,real_label=getPredLabel(FilePath)
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
	# dot=1.0
	nowTime=time.strftime(ISOTIMEFORMAT,time.localtime())
	f=open(log_path,'a')
	f.write('--------seed:%d--time:%s-------------\n' %(seed,nowTime))
	for i in range(class_count):
		precision[i]+=(tp[i]/(tp[i]+fp[i]))
		recall[i]+=(tp[i]/p[i])
		# dot*=recall[i]
		if precision[i]+recall[i]!=0:
			F1[i]+=((2*precision[i]*recall[i])/(precision[i]+recall[i]))
		else:
			F1[i]=0
		f.write('label:%d \t precision:%.5f \t recall:%.5f \t F1:%.5f\n' %(i,precision[i],recall[i],F1[i]))
	f.write('------------acc:%.5f-----------------\n\n' %(acc))
	f.close()
	# print('acc:%.5f' %(acc))
	return acc

def createPRF_me(pred_label,real_label,law_index,lenTP,lenTN,lenP,lenN,log_path):  # 计算prf值 写入文件  rate ig 比率
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
	# dot=1.0
	nowTime=time.strftime(ISOTIMEFORMAT,time.localtime())
	f=open(log_path,'a')
	f.write('--law_index:%d--lenTP:%d--lenTN:%d--lenP:%d--lenN:%d--raw_rate:%.3f--time:%s-\n' \
			%(law_index,lenTP,lenTN,lenP,lenN,max(lenP,lenN)/(lenP+lenN),nowTime))
	for i in range(class_count):
		if tp[i]+fp[i]!=0:
			precision[i]+=(tp[i]/(tp[i]+fp[i]))
		else:
			precision[i]=0
		if p[i]!=0:
			recall[i]+=(tp[i]/p[i])
		else:
			recall[i]=0
		# dot*=recall[i]
		if precision[i]+recall[i]!=0:
			F1[i]+=((2*precision[i]*recall[i])/(precision[i]+recall[i]))
		else:
			F1[i]=0
		f.write('label:%d \t precision:%.5f \t recall:%.5f \t F1:%.5f\n' %(i,precision[i],recall[i],F1[i]))
	f.write('------------acc:%.5f-----------------\n\n' %(acc))
	f.close()
	# print('acc:%.5f' %(acc))
	return acc

def getPredProb(FilePath):
	pred_prob=[]
	real_label=[]
	pred_label=[]
	f=open(FilePath,'rb')
	for line in f:
		pieces=line.strip().split()
		real_l=pieces[1]
		pred_l=pieces[2].split(':')[0]
		pred_label.append(pred_l)
		real_label.append(real_l)
		probDict={}
		pieces=pieces[2:]
		for item in pieces:
			p=item.strip().split(':')
			label=int(p[0])
			prob=float(p[1])
			probDict[label]=prob
		pred_prob.append(probDict)
	# print pred_prob[0][0]
	return pred_prob,pred_label,real_label

def me_classify(trains,tests,result_file='result.txt'):  # 默认为result.txt
	# lexcion=get_lexcion(trains)
	lexcion=getDFWords(trains)
	print('len(lexcion):'+str(len(lexcion)))
	createFormatText(trains,lexcion,'train.txt')
	createFormatText(tests,lexcion,'test.txt')
	file2Bin('train.txt','train.bin')
	file2Bin('test.txt','test.bin')
	train('train.bin','train.model')
	classify('train.model','test.bin',result_file)

# getPredLabel('result.txt')
# createPRF('result.txt')



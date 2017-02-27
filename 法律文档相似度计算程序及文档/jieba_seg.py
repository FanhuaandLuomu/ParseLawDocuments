#coding:utf-8
# 使用结巴分词对语料分词
import jieba
import jieba.posseg as pseg
# import threading
import sys
import os
import time

def cut(filename1,filename2): # 对每一个文件seg and pos
	f=open(filename2,'w')
	for line in open(filename1):
		res=pseg.cut(line.strip())
		split_line=' '.join([w.word for w in res])+'\n'
		f.write(split_line.encode('utf-8'))
	# print '%s split successful' %(filename1)

def getFileList(source_path,target_path):
	if not os.path.exists(target_path):
		os.makedirs(target_path)

	filenames=os.listdir(source_path)
	source_files=[source_path+os.sep+filename for filename in filenames]
	target_files=[target_path+os.sep+filename for filename in filenames]
	return source_files,target_files

def main():
	source_path,target_path=sys.argv[1],sys.argv[2]
	source_files,target_files=getFileList(source_path,target_path)
	# print fileList
	for filename1,filename2 in zip(source_files,target_files):
		cut(filename1,filename2)

if __name__ == '__main__':
	t1=time.time()
	main()
	t2=time.time()
	print 'seg success.cost:%.2f' %(t2-t1)
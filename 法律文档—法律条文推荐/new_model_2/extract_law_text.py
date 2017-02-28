#coding:utf-8
import os
import cPickle
import jieba
import jieba.posseg as pseg

def readFromFile(filename):  # 读文件
	law_text_list=[]
	with open(filename,'r') as f:
		for line in f:
			pieces=line.decode('GB18030').strip().split('===')
			law,text=pieces[0],pieces[1]
			law_text_list.append(text.strip())
	return law_text_list

def cut(contents):  # 分词
	split_contents=[]
	for line in contents:
		res=pseg.cut(line.strip())
		split_line=' '.join([w.word for w in res])
		split_contents.append(split_line)
	return split_contents

def main():
	source_file='law_text.txt'
	law_text_list=readFromFile(source_file)
	print len(law_text_list)

	split_contents=cut(law_text_list)
	# cPickle.dump(split_contents,open('split_law_text.pkl','wb'))
	print len(split_contents)

	# for item in law_text_list:
	# 	print item

	print law_text_list[1].strip()
	print split_contents[1].strip()

if __name__ == '__main__':
	main()
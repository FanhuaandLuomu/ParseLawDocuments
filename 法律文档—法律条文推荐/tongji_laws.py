#coding:utf-8
# 从caipanyiju.txt裁判依据中提取所涉及的法律条文
# 写入文件law_pieces.txt中   /更新：删除了一些不好的案件 重复、没有事实的一些==> law_pieces2.txt
# 同时保存所有案件涉及的法律条文  laws_list.pkl    laws_list2.pkl
import os
import re
import cchardet
import cPickle

def get_all_yiju(filename):  # 得到所有的裁判依据
	yiju_list=[]
	f=open(filename,'r')
	lines=f.readlines()
	for i,line in enumerate(lines):
		if (i+1)%4==0:  # i=3,7,11...
			yiju_list.append(lines[i].decode('utf-8'))
	return yiju_list

def deal_yiju(yiju):  # 处理法律依据 提取法律条文  xx法xx条
	laws=[]
	# match1=re.finditer(u'(最高人民法院)?(《.*?》)第',yiju)
	match1=re.finditer(u'(最高人民法院.{0,10})?《[^，]+?》(若干问题的解释)?(若干问题的意见)?(（二）)?第',yiju)
	for item in match1:
		laws.append([item.group()[:-1],item.start()])
	laws.append(['None',len(yiju)-1])
	# for item in laws:
	# 	print item[0],item[1]

	pieces=[]
	match2=re.finditer(u'第[^《、款]{1,6}条',yiju)  # 有的法律条文为空 不满足正则...
	for item in match2:
		pieces.append([item.group(),item.start()])
	# for item in pieces:
	# 	print item[0],item[1]

	law_pieces=[]
	for item in pieces:
		p,start1=item[0],item[1]
		for i in range(len(laws)-1):
			if laws[i][1]<start1<laws[i+1][1]:
				if [laws[i][0],p] not in law_pieces:  # 每个案件中 xx法xx条 唯一化
					law_pieces.append([laws[i][0],p])
				break
	return law_pieces

def write2file(contents,yiju_list,filename):
	f=open(filename,'w')
	for i,law_pieces in enumerate(contents):
		f.write(yiju_list[i].encode('utf-8')+'\n')
		if len(law_pieces)==0:
			f.write('None\n')
		for item in law_pieces:
			law,piece=item[0],item[1]
			f.write('%s-%s\n' %(law.encode('utf-8'),piece.encode('utf-8')))
		f.write('='*20+'\n')
	f.close()

def main():
	source_file='caipanyiju_new.txt'  # 从文件中读取数据
	law_piece_file='law_pieces_new.txt'
	yiju_list=get_all_yiju(source_file)
	print len(yiju_list)
	laws_list=[]
	for yiju in yiju_list:
		law_pieces=deal_yiju(yiju)   # 一个案件涉及的法律条文[p1,p2,...]
		laws_list.append(law_pieces)
	# laws_list 存储12192个案件所涉及的法律条文  去重后12043个案件
	write2file(laws_list,yiju_list,law_piece_file) 
	# laws_list: [item1,...]    item1:[[law,piece],[],...] 一个案件所涉及的法律条文
	cPickle.dump(laws_list,open('laws_list_new.pkl','wb'))

if __name__ == '__main__':
	main()
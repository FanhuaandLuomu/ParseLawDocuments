#coding:utf-8
# 暂时使用结巴分词 进行中文分词
# ../contents.pkl  -->  split_contents.pkl  12043个
import jieba
import jieba.posseg as pseg
import cPickle
import cchardet

def cut(contents):  # 分词
	split_contents=[]
	for line in contents:
		res=pseg.cut(line.strip())
		split_line=' '.join([w.word for w in res])
		split_contents.append(split_line)
	return split_contents

def main():
	# [item1,item2,...]  item 案件的首部+事实部分 序号即为dict中的id 共有10625个案件
	contents=cPickle.load(open('../contents_new.pkl','rb'))
	print contents[0]
	split_contents=cut(contents)
	print split_contents[0]
	cPickle.dump(split_contents,open('split_contents_new.pkl','wb'))

if __name__ == '__main__':
	main()
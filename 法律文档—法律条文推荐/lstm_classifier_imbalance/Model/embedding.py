#coding:utf-8
import cPickle
import gensim

class MySentences(object):
	def __init__(self,contents):
		self.contents=contents
	def __iter__(self):
		for line in self.contents:
			words=[]
			for word in line.strip().split():
				words.append(word)
			yield words

def main():
	# 首先载入数据 10672个
	contents=cPickle.load(open('../split_contents_new.pkl','rb'))
	print len(contents)
	sentences=MySentences(contents)
	model=gensim.models.Word2Vec(sentences,size=100,window=5,min_count=3,workers=4,iter=10)
	# 保存模型
	model.save('model.m')
	model.save_word2vec_format('model.txt',binary=False)

	# 载入model
	# model=gensim.models.Word2Vec.load('model.m')

if __name__ == '__main__':
	main()
#coding:utf-8
import cPickle

def get_count(count,rate):
	k=0
	for c in count:
		if c>rate:
			k+=1
	return k

def main():
	new_contents=cPickle.load(open('split_contents_new.pkl'))
	count=[]
	for item in new_contents[:1000]:
		count.append(len(item.split()))
	k=get_count(count,1000)
	print k

if __name__ == '__main__':
	main()
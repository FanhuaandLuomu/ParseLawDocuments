#coding:utf-8
# 从tags_punct_case_cut(已按模块划分)中读取数据
# 实验中提取每个案件的首部+事实+裁判依据
# 根据一些规范去掉了一些不好的样本dirty.txt,清洗数据
# eg。首部为空则直接去掉，首部不为空看事实是否为空,不为空保留，为空再根据关键字筛选。
# 生成experiment/caipanyiju.txt文件
# experiment/contents_new.pkl
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import cPickle

def readFromFile(filename):  # 读文件
	contents=[]
	lines=open(filename,'r').readlines()
	for i,line in enumerate(lines):
		line=line.strip().decode('utf-8','ignore')
		contents.append(line)
	return contents

def get_info(contents):  # 提取每个案件的首部 事实信息 和 裁判依据
	text1=[]  # 案件首部
	text2=[]
	text3=[]  # 裁判依据
	for i,line in enumerate(contents):
		if line=='#首部':      # 可添加 裁判理由（本院认为）
			text1.append(contents[i+1])
		if line=='#事实':
			text2.append(contents[i+1])
		if line=='#裁判依据':
			text3.append(contents[i+1])
	return '\n'.join(text1),'\n'.join(text2),'\n'.join(text3)

def write2file(fname,res_list):  # 写入文件
	f=open(fname,'w')
	for i,res in enumerate(res_list):
		f.write('-----%d-----\n' %(i))
		f.write('%s\n%s\n%s\n' %(res[0],res[1],res[2]))
	f.close()

def main():
	source_path='tags_punct_case_cut'
	# source_path='tmp_case_cut'
	target_path='experiment'
	saved_file='caipanyiju_new.txt'
	filenames=os.listdir(source_path)
	print len(filenames)
	# print filenames[0]
	f2=open('dirty.txt','w')   # 保存舍弃的案件
	res_list=[]
	differ_contents=[]  # 供去重使用
	for i,filename in enumerate(filenames):
		contents=readFromFile(source_path+os.sep+filename)
		text1,text2,text3=get_info(contents)  # 首部 事实 裁判依据
		if text1.strip()!='':  # 首部必须不为空
			if text2.strip()!='':  # 事实不为空  直接保留
				if text1.strip()+text2.strip() not in differ_contents:
					res_list.append([text1,text2,text3])
					differ_contents.append(text1.strip()+text2.strip())
			# elif u'撤诉' not in text1 or u'撤回' not in text1 or len(text1)>=80:  # 事实为空  判断首部 分情况
			# 	res_list.append([text1,text2,text3])
			# 	differ_contents.append(text1.strip()+text2.strip())
			# 关键字匹配  去掉一些信息量几乎没有的案件
			else:  # 事实为空的情况 考虑首部	
				if u'文素。' in text1:   # 通过关键字去掉一些不好的案件
					print '='*20
					f2.write(text1+'\n')
					continue			
				if u'撤诉' in text1 or u'撤回' in text1 or u'担保' in text1 or u'审理' in text1 or\
					u'保全' in text1 or u'合同' in text1 or u'本院' in text1 :  # 事实为空  判断首部 分情况
					if text1.strip()+text2.strip() not in differ_contents:
						res_list.append([text1,text2,text3])
						differ_contents.append(text1.strip()+text2.strip())
				else:  # 事实为空 且不包含一些应用的关键字  去掉
					f2.write(text1+'\n')
		# if text1.strip()!='' and text2.strip()=='':  #首部不为空  事实为空
		# 	if u'撤诉' not in text1 and u'撤回' not in text1 and u'担保' not in text1 and u'审理' not in text1 and\
		# 		u'保全' not in text1 and u'合同' not in text1 and u'本院' not in text1:
				# f2.write(text1+'\n')
		print i
	f2.close()
	print 'len(res_list):',len(res_list)
	print 'len(differ_contents):',len(differ_contents)
	# differ_contents  存放有用的案件的首部+事实  顺序很重要 与案件对应的案件号一致
	cPickle.dump(differ_contents,open(target_path+os.sep+'contents_new.pkl','wb'))
	write2file(target_path+os.sep+saved_file,res_list)
	

if __name__ == '__main__':
	main()
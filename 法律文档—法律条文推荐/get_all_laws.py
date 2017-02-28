#coding:utf-8
# 从laws_list.pkl/laws_list2.pkl中提取xx法xx条 及 涉及该法律条文的案件号
# 最终得到new_all_pieces_count_dict.pkl    /   2
#         new_all_laws_count_dict.pkl      /   2
import cPickle
import cchardet
import re

def merge_laws(laws_list):  # key:xx法xx条   value:包含该法律条文的案件号
	all_laws=[]
	all_laws_dict={}
	all_pieces=[]
	all_pieces_dict={}
	for i,law_pieces in enumerate(laws_list):  # i 为案件的编号
		tmp_law=[]
		# if i==442:
		# 	print law_pieces
		for item in law_pieces:  # law_pieces中无重复项
			law,piece=item[0],item[1] # xx法律  xx条
			# 一个案件出现多个xx法只算一次
			if law not in tmp_law:  # tmp_law存放该案件中出现的法律
				tmp_law.append(law)

			if law not in all_laws:
				all_laws.append(law)

			if law+'-'+piece not in all_pieces:
				all_pieces.append(law+'-'+piece)
				all_pieces_dict[law+'-'+piece]=[i]
			else:
				all_pieces_dict[law+'-'+piece].append(i)
		for law in tmp_law:  # 引入tmp_law是为了解决一个案件中出现多个相同xx法的情况 
			if law not in all_laws_dict:
				all_laws_dict[law]=[i]
			else:
				all_laws_dict[law].append(i)
	return all_laws,all_laws_dict,all_pieces,all_pieces_dict

def digit2chinese(digit):  # 将数字转换为汉字  最多考虑百位
	tmp_list = [u'零', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九']
	# print len(digit)
	if len(digit)==1:
		res=tmp_list[int(digit)]
		return res
	if len(digit)==2:
		res=tmp_list[int(digit[0])]+u'十'+tmp_list[int(digit[1])]
		res=res.replace(u'十零',u'十').replace(u'一十',u'十')
		return res
	if len(digit)==3:
		res=tmp_list[int(digit[0])]+u'百'+tmp_list[int(digit[1])]+u'十'+\
			tmp_list[int(digit[2])]
		res=res.replace(u'零十',u'零').replace(u'十零',u'十').replace(u'零零',u'')
		return res
	return digit
				
def remove_punct(piece):  # 法律条文处理
	punct=[u'《',u'》',u'﹤',u'﹥',u'（',u'）',u'〈',u'〉',u'＜',u'＞',u'、',u'[',u'。',\
			u'｛',u'｝',u'?']
	for c in punct:
		piece=piece.replace(c,'')
	# 处理一些特殊法律条文
	pieces=piece.split('-')
	if pieces[0] in [u'最高人民法院关于适用中华人民共和国婚姻法司法解释二',\
			     u'最高人民法院中华人民共和国婚姻法解释二',\
			     u'中华人民共和国婚姻法若干问题的解释二',\
			     u'关于适用中华人民共和国婚姻法若干问题的解释二']:
		piece=u'最高人民法院关于适用中华人民共和国婚姻法若干问题的解释二'+'-'+pieces[1]

	if pieces[0] in [u'婚姻法']:
		print '---'
		piece=u'中华人民共和国婚姻法'+'-'+pieces[1]
	if pieces[0] in [u'合同法']:
		piece=u'中华人民共和国合同法'+'-'+pieces[1]
	if pieces[0] in [u'民事诉讼法']:
		piece=u'中华人民共和国民事诉讼法'+'-'+pieces[1]
	if pieces[0] in [u'民法通则']:
		piece=u'中华人民共和国民法通则'+'-'+pieces[1]
	if pieces[0] in [u'担保法']:
		piece=u'中华人民共和国担保法'+'-'+pieces[1]
	if pieces[0] in [u'物权法']:
		piece=u'中华人民共和国物权法'+'-'+pieces[1]
	if pieces[0] in [u'继承法']:
		piece=u'中华人民共和国继承法'+'-'+pieces[1]
	if pieces[0] in [u'人民法院诉讼收费办法']:
		piece=u'最高人民法院人民法院诉讼收费办法'+'-'+pieces[1]

	if piece[-1]==piece[-2]==u'条':  # xx条条的情况
		piece=piece[:-1]

	if piece[:3]==u'第第第':
		piece=piece[3:]

	if u'-第第' in piece:
		piece=piece.replace(u'-第第',u'-第')

	if piece[:2]==u'第第':
		piece=piece[2:]

	key=re.findall(u'-(第.*?条)法律条',piece)
	if len(key)>0:
		key=key[0]
		pieces=piece.split('-')
		piece=pieces[0]+'-'+key

	key=re.findall(u'第(.*?)条',piece)  # 判断是否为数字 若是则转为汉字表示
	if len(key)>0 and key[0].isdigit():
		num=key[0]
		pieces=piece.split(num)
		chars=digit2chinese(num)
		piece=pieces[0]+chars+pieces[1]

	if u'借贷案件' in piece and u'意见' in piece:
		pieces=piece.split('-')
		piece=u'最高人民法院关于审理借贷案件的若干意见'+'-'+pieces[1]
	tmp=re.findall(u'中[华国].*?国',piece)
	if len(tmp)>0:
		tmp=tmp[0]
		piece=piece.replace(tmp,u'中华人民共和国')
	if u'最高人民法院关于民事诉讼证据若干规定' in piece:
		pieces=piece.split('-')
		piece=u'最高人民法院关于民事诉讼证据的若干规定'+'-'+pieces[1]
	return piece

def merge_same_laws(all_pieces_count_dict):  # 合并相同的法律条文
	new_all_pieces_count_dict={}
	for key in all_pieces_count_dict:
		key2=remove_punct(key)   # 去除标点 相似的法律条文合并
		# if key2==u'最高人民法院关于适用中华人民共和国民事诉讼法的解释-第二百一十三条':
		# 	print key,all_pieces_count_dict[key][0]
		if key2 not in new_all_pieces_count_dict:  # key2不在，则直接赋值
			new_all_pieces_count_dict[key2]=all_pieces_count_dict[key]
		else:
			new_all_pieces_count_dict[key2][1]=list(set((new_all_pieces_count_dict[key2][1]+\
								all_pieces_count_dict[key][1])))
			new_all_pieces_count_dict[key2][0]=len(new_all_pieces_count_dict[key2][1])
		# 涉及key2法律条文的案件按案件id顺序排序
		new_all_pieces_count_dict[key2][1]=sorted(new_all_pieces_count_dict[key2][1])
	return new_all_pieces_count_dict

# 得到每个案件的（按序号）相关法律条文
def get_each_case_laws(sid,new_all_pieces_count_dict):
	pieces=[]
	for item in new_all_pieces_count_dict.items():
		if sid in item[1][1]:
			pieces.append(item[0])
	return pieces

# k为阈值 得到【不涉及】出现次数小于等于k的法律条文的案件 
# 若 xx法xx条出现次数<=k，则将出现该条的案件都去掉
def get_nums_by_del_some_pieces(k,new_all_pieces_count_dict):
	nums=range(10672)
	# 按法律出现次数从小到大排序
	for item in sorted(new_all_pieces_count_dict.items(),key=lambda x:x[1]):
		if item[1][0]<=k:
			for c in item[1][1]:
				if c in nums:
					nums.remove(c)
		else:
			break
	return nums

def get_counts_of_laws(new_all_pieces_count_dict):  # 统计xx法出现的情况
	new_all_laws_count_dict={}
	for key in new_all_pieces_count_dict:  # key xx法xx条
		law=key.split(u'-')[0]   # xx法
		ilist=new_all_pieces_count_dict[key][1]
		if law not in new_all_laws_count_dict:  # 第一次出现
			new_all_laws_count_dict[law]=new_all_pieces_count_dict[key]
		else:   # 前面已经出现过
			new_all_laws_count_dict[law][1]=list(set((new_all_laws_count_dict[law][1]+\
									ilist)))
			new_all_laws_count_dict[law][0]=len(new_all_laws_count_dict[law][1])
		# 按 案件 id顺序排序
		new_all_laws_count_dict[law][1]=sorted(new_all_laws_count_dict[law][1])
	return new_all_laws_count_dict

def write2file(idict,filename):  # 将laws和pieces的字典写入文件
	# 写入文件
	f=open(filename,'w')
	for item1 in sorted(idict.items(),key=lambda x:x[1][0],reverse=True):
		f.write(item1[0].encode('utf-8','ignore')+'\t'+str(item1[1][0])+'\n')
	f.close()

def main():
	source_file='laws_list_new.pkl'
	# laws_list: [[[law,piece],[],...],...]
	laws_list=cPickle.load(open(source_file,'rb'))
	print 'len(laws_list):',len(laws_list)
	# all_laws_dict 中key为xx法xx条   value 为涉及该法律的案件
	all_laws,all_laws_dict,all_pieces,all_pieces_dict=merge_laws(laws_list)
	print 'len(all_laws):',len(all_laws)
	print 'len(all_pieces):',len(all_pieces)
	# all_laws_count_dict 中 key为xx法xx条 value 为 [涉及该法律的案件数count,[案件号...]]
	all_laws_count_dict={}
	for key in all_laws_dict:
		all_laws_count_dict[key]=[len(all_laws_dict[key]),all_laws_dict[key]]

	# all_pieces_count_dict 中 key为xx法xx条 value为 [涉及该法律的案件数count,[涉及该法律的案件编号...]]
	all_pieces_count_dict={}
	for key in all_pieces_dict:
		all_pieces_count_dict[key]=[len(all_pieces_dict[key]),all_pieces_dict[key]]

	# cPickle.dump(all_laws_count_dict,open('all_laws_count_dict.pkl','wb'))
	# cPickle.dump(all_pieces_count_dict,open('all_pieces_count_dict.pkl','wb'))

	# f=open('all_laws.txt','w')
	# for item in sorted(all_laws_count_dict.items(),key=lambda x:x[1],reverse=True):
	# 	f.write(item[0].encode('utf-8','ignore')+'\t'+str(item[1][0])+'\n')
	# f.close()
	# f=open('all_pieces.txt','w')
	# for item in sorted(all_pieces_count_dict.items(),key=lambda x:x[1],reverse=True):
	# 	f.write(item[0].encode('utf-8','ignore')+'\t'+str(item[1][0])+'\n')
	# f.close()
	# print all_pieces_count_dict[u'《中华人民共和国合同法》-第二百零七条'][0]
	# print all_laws_count_dict[u'最高人民法院关于适用《中华人民共和国担保法》若干问题的解释》']

	# 一、{xx法xx条:[涉及该法的案件数量,[案件的编号...]],...}
	new_all_pieces_count_dict=merge_same_laws(all_pieces_count_dict)
	print 'len(new_all_pieces_count_dict):',len(new_all_pieces_count_dict)
	# 【保存】
	cPickle.dump(new_all_pieces_count_dict,open('new_all_pieces_count_dict_new.pkl','wb'))
	# print new_all_pieces_count_dict[u'中华人民共和国合同法-第二百零六条'][1]
	# print new_all_pieces_count_dict[u'中华人民共和国合同法-第二百零六条'][0]
	# 写入文件
	write2file(new_all_pieces_count_dict,'new_all_pieces_count_dict_new.txt')

	# 得到某案件出现的法律条文 xx法xx条  14为案件编号
	pieces=get_each_case_laws(14,new_all_pieces_count_dict)
	for item in pieces:
		print item

	# 若 xx法xx条出现次数<=k，则将出现该条的案件都去掉,有241条案件不涉及相关法律或正则匹配失败
	nums=get_nums_by_del_some_pieces(1,new_all_pieces_count_dict)
	print len(nums)	
	# print nums

	# 二、[xx法:[涉及该法的案件数量,[案件的编号...]]]   此时 new_all_pieces_count_dict 发生了变化
	new_all_laws_count_dict=get_counts_of_laws(new_all_pieces_count_dict)
	print 'len(new_all_laws_count_dict):',len(new_all_laws_count_dict)
	# 【保存】
	cPickle.dump(new_all_laws_count_dict,open('new_all_laws_count_dict_new.pkl','wb'))
	# print new_all_laws_count_dict[u'最高人民法院关于适用中华人民共和国婚姻法若干问题的解释二'][0]
	# print new_all_laws_count_dict[u'最高人民法院关于适用中华人民共和国婚姻法若干问题的解释二'][1]
	# 写入文件
	write2file(new_all_laws_count_dict,'new_all_laws_count_dict_new.txt')

	# 得到某案件出现的法律 xx法
	pieces=get_each_case_laws(339,new_all_laws_count_dict)
	for item in pieces:
		print item

if __name__ == '__main__':
	main()
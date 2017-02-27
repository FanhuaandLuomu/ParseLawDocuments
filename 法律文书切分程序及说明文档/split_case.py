#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import re

def get_text(filename):
	f=open(filename,'r')
	lines=[]
	for line in f:
		line=line.replace(' ','').replace(r'\t','')
		if '<style' in line or 'TABLE' in line:  # 个别样本有噪音
			continue
		line=line.strip().decode('GB18030')
		if u'据此' in line and u'《' in line:
			index=line.index(u'据此')
		elif u'依照《' in line and u'《' in line:
			index=line.index(u'依照《')
		elif u'根据《' in line and u'《' in line:
			index=line.index(u'根据《')
		elif u'依据《' in line and u'《' in line:
			index=line.index(u'依据《')
		else:
			index=-1
		if index!=-1:
			line1=line[:index]
			line2=line[index:]
			lines.append(line1)
			lines.append(line2)
		else:
			lines.append(line)
	return lines

def find_line_no(lines, s, flag):  # 根据关键字s在lines中寻找对应的行  flag False 倒序寻找
	line_no = -1
	for i, line in enumerate(lines):
		try:
			line.index(s)
			line_no = i
			if s in [u'审判杨刚',u'院长吴颖', u'执行员曹海燕',u'审判长', u'审判员',u'审理员',u'陪审员'] and len(line) > 20:#长度大于20，找到的不算
				line_no = -1
				continue

			if flag:
				return line_no
		except:
			continue
	return line_no #-1表示没找到

def cut_case(filename):  # 按规则切分文本
	print filename
	lines=get_text(filename)
	#  标题
	title={}   
	if len(lines)<3:
		return []
	title['fayuan_name']=lines[0]
	title['wenshu_name']=lines[1]
	title['anhao']=lines[2]

	# 落款
	luokuan={}
	luokuan_begin_no = find_line_no(lines, u'院长', True)
	if luokuan_begin_no == -1:
		luokuan_begin_no = find_line_no(lines, u'审判长', True)
		if luokuan_begin_no==-1:
			luokuan_begin_no=find_line_no(lines,u'审判杨刚',True)
		if luokuan_begin_no==-1:
			luokuan_begin_no=find_line_no(lines,u'执行员',True)
		if luokuan_begin_no==-1:
			luokuan_begin_no=find_line_no(lines,u'陪审员',True)
		if luokuan_begin_no==-1:
			luokuan_begin_no=find_line_no(lines,u'审理员',True)
		if luokuan_begin_no==-1:

			luokuan_begin_no=find_line_no(lines,u'审判员',True)

			# print len(lines[luokuan_begin_no])
			# print lines[luokuan_begin_no]
	luokuan_end_no = find_line_no(lines[luokuan_begin_no:], u'书记员', False)
	if luokuan_end_no!=-1:
		luokuan_end_no+=luokuan_begin_no
	print luokuan_begin_no,luokuan_end_no
	# lines=lines[:luokuan_end_no+1]
	date1=find_line_no(lines[luokuan_begin_no:luokuan_end_no+1],u'年',False)
	date2=find_line_no(lines[luokuan_begin_no:luokuan_end_no+1],u'月',False)
	date3=find_line_no(lines[luokuan_begin_no:luokuan_end_no+1],u'日',False)
	# print date1,date2,date3
	if date1==date2==date3!=-1:
		date=lines[luokuan_begin_no:luokuan_end_no+1][date1]
	else:
		date=''
	shuming = ';'.join(lines[luokuan_begin_no: luokuan_begin_no+date1]+lines[luokuan_begin_no+date1+1:luokuan_end_no+1])
	luokuan['date']=date
	luokuan['shuming']=shuming

	# 正文
	main_content={}
	# ==尾部
	match = re.search(ur'(受理|诉讼|减半)(.{,20})(\d+元|免收)', ''.join(lines[:luokuan_begin_no][::-1]))
	if match:
		s = match.group()
		weibu_begin_no = find_line_no(lines[:luokuan_end_no+1], s, False) #尾部第一行行号
		weibu=''.join(lines[weibu_begin_no:luokuan_begin_no])
	else:
		# print 'yes'
		if u'上述' in lines[luokuan_begin_no-1]:
			weibu_begin_no=luokuan_begin_no-1
		else:
			weibu_begin_no=luokuan_begin_no
			weibu=''.join(lines[weibu_begin_no:luokuan_begin_no])
	# print weibu_begin_no
	main_content['weibu']=weibu
	#==裁判主文
	caipan_begin_no=find_line_no(lines[:weibu_begin_no],u'判决如下',False)
	if caipan_begin_no==-1:
		caipan_begin_no=find_line_no(lines[:weibu_begin_no],u'裁定如下',False)
		
	if caipan_begin_no==-1:
		caipan_begin_no=find_line_no(lines[:weibu_begin_no],u'本院认为',False)
		if caipan_begin_no!=-1:
			caipan_begin_no+=1
	if caipan_begin_no==-1:
		caipan_zhuwen=''
	else:
		caipan_zhuwen=''.join(lines[caipan_begin_no+1:weibu_begin_no])
	main_content['caipan_zhuwen']=caipan_zhuwen
	#==裁判依据
	caipan_yiju=lines[caipan_begin_no]
	main_content['caipan_yiju']=caipan_yiju
	#==理由
	liyou=lines[caipan_begin_no-1]
	main_content['liyou']=liyou
	#==首部
	shoubu_end_no=find_line_no(lines[:luokuan_end_no+1],u'审理终结。',True)
	if shoubu_end_no==-1:
		shoubu_end_no=find_line_no(lines[:luokuan_end_no+1],u'撤诉申请。',True)
		if shoubu_end_no==-1:
			shoubu_end_no=find_line_no(lines[:luokuan_end_no+1],u'诉称：',True)
			if shoubu_end_no!=-1:
				shoubu_end_no-=1
	if shoubu_end_no>=caipan_yiju or shoubu_end_no==-1:
		shoubu_end_no=caipan_begin_no-2
	shoubu=''.join(lines[3:shoubu_end_no+1])
	main_content['shoubu']=shoubu
	#==事实
	shishi=''.join(lines[shoubu_end_no+1:caipan_begin_no-1])
	main_content['shishi']=shishi
	# print filename
	return [title,main_content,luokuan]

def write2file_cut(target_path,item):
	filename=item[0]
	contents=item[1]   # [title,zhengwen,luokuan]
	title=contents[0]  # title
	main_content=contents[1]  # main
	luokuan=contents[2]  # luokuan
	fayuan_name,wenshu_name,anhao=title['fayuan_name'],title['wenshu_name'],title['anhao']
	shoubu=main_content['shoubu']
	shishi=main_content['shishi']
	liyou=main_content['liyou']
	caipan_yiju=main_content['caipan_yiju']
	caipan_zhuwen=main_content['caipan_zhuwen']
	weibu=main_content['weibu']
	date,shuming=luokuan['date'],luokuan['shuming']

	contents=[fayuan_name,wenshu_name,anhao,shoubu,shishi,liyou,caipan_yiju,caipan_zhuwen,weibu,
					shuming,date]

	f=open(target_path+os.sep+filename,'w')
	f.write('===标题===\n')
	f.write('#法院名称\n')
	f.write(contents[0]+'\n')
	f.write('#文书名称\n')
	f.write(contents[1]+'\n')
	f.write('#案号\n')
	f.write(contents[2]+'\n')

	f.write('===正文===\n')
	f.write('#首部\n')
	f.write(contents[3]+'\n')
	f.write('#事实\n')
	f.write(contents[4]+'\n')
	f.write('#理由\n')
	f.write(contents[5]+'\n')
	f.write('#裁判依据\n')
	f.write(contents[6]+'\n')
	f.write('#裁判主文\n')
	f.write(contents[7]+'\n')
	f.write('#尾部\n')
	f.write(contents[8]+'\n')

	f.write('===落款===\n')
	f.write('#署名\n')
	f.write(contents[9]+'\n')
	f.write('#日期\n')
	f.write(contents[10]+'\n')
	f.close()

def main(source_path,target_path):
	if not os.path.exists(target_path):
		os.mkdir(target_path)
	case_list=[]
	filenames=os.listdir(source_path)
	for filename in filenames:
		contents=cut_case(source_path+os.sep+filename)
		if len(contents)>0:
			case_list.append([filename.decode('GB18030'),contents])
	for item in case_list:
		write2file_cut(target_path,item)
	print len(case_list)
	# print error_count

if __name__ == '__main__':
	source_path,target_path=sys.argv[1],sys.argv[2]
	main(source_path,target_path)
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(FILE_PATH, "../..")
DATA_PATH = os.path.join(ROOT_PATH, "data")

def get_num(data_name):
	path=os.path.join(DATA_PATH,data_name,"label.txt")
	file=open(path,'r',encoding='gb2312')
	lines=file.readlines()
	file.close()
	return len(lines)

def load_label(data_name):
    node_path=os.path.join(DATA_PATH,data_name,"node.txt")
    node_file=open(node_path,'r')
    nodes=node_file.readlines()
    node_num=len(nodes)
    node_file.close()
    return node_num

def load_embedding(data_name,node_num):
	embed_file=data_name+ "_deepwalk_emb.dat"
	file=open(embed_file)
	lines = file.readlines()
	file.close()
	line_num=len(lines)
	node_number,dim=lines[0].split()
	#node_number=int(node_number)
	dim=int(dim)
	embedding=np.zeros((node_num,dim))
	for i in range(1,line_num):
	    x=lines[i].split()
	    embedding[int(x[0]),:]=list(map(float,x[1:]))
	return embedding

def load_dic(data_path):
	# load mus.dat  or sigs.dat
	file=open(data_path,'r')
	data=file.readlines()
	file.close()
	num=len(data)
	dic={}
	for line in data:
		items=line.split('\t')
		emb=items[1].split()
		dim=len(emb)
		dic[items[0]]=list(map(float,emb))
	return dic

def load_num2name(data_name):
	path=os.path.join(DATA_PATH,data_name,"tag.txt")
	file=open(path,'r',encoding='gb2312')
	lines=file.readlines()
	file.close()

	num2name={}
	for line in lines:
		items=line.split()
		num=int(items[-1])
		name=' '.join(items[:-1])
		num2name[num]=name

	return num2name


def generate_dist_emb(mus_dic,sig_dic,data_name,num2name):
	# load label
	label_path=os.path.join(DATA_PATH,data_name,"tag-edge.txt")
	file=open(label_path,'r',encoding='gb2312')
	labels=file.readlines()
	file.close()
	node_num=len(labels)

	for i in range(node_num):
		node,tag = labels[i].strip('\n').split()
		labels[i]=int(labels[i].strip('\n').split())

	# generate dist embedding
	dist_emb=[] 
	for i in range(node_num):
		mus_ = mus_dic[num2name[labels[i]]]
		sig_ = sig_dic[num2name[labels[i]]]
		tmp=np.append(mus_,sig_)
		dist_emb.append(tmp)

	return np.array(dist_emb, dtype=np.float32)

def joint(embedding1,embedding2):
	final_embedding=np.append(embedding1,embedding2,axis=1)
	return final_embedding,len(final_embedding[0])


def write_file(embedding,data_name,node_number,dim):
	path=data_name+"_deepwalk_dist_emb.dat"
	file=open(path,'w')
	file.writelines([str(node_number),' ',str(dim),'\n'])
	for i in range(node_number):
		file.writelines(str(i))
		for j in range(dim):
			file.writelines([' ',str(embedding[i][j])])
		file.write('\n')
	file.close()

def main():
	
	data=['leetcode','bilibili1','bilibili2']
	for data_ in data:
		node_num=get_num(data_)
		embedding=load_embedding(data_,node_num)
		
		mus_path=os.path.join(data_,'mus.dat')
		mus_dic=load_dic(mus_path)
		sig_path=os.path.join(data_,'sigs.dat')
		sig_dic=load_dic(sig_path)

		num2name=load_num2name(data_)

		dist_emb=generate_dist_emb(mus_dic,sig_dic,data_,num2name)
		final_embed,dim=joint(embedding,dist_emb)
		print("dim =",dim)
		write_file(final_embed,data_,node_num,dim)



if __name__ == '__main__':
	main()
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

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

def load_one_hot(data_name,node_number):
    path=os.path.join(DATA_PATH,data_name,"tag-edge.txt")
    file=open(path,'r',encoding='gb2312')
    lines=file.readlines()
    file.close()
    node_num=len(lines)
    l=[[] for i in range(node_number)]
    
    for i in range(node_num):
    	items=lines[i].split()
    	entity=int(items[0])
    	tag=int(items[1])
    	l[entity].append(tag)
        
    mlb = MultiLabelBinarizer()
    one_hot_embed = mlb.fit_transform(l)
    return one_hot_embed, len(one_hot_embed[0])

def joint(embedding1,embedding2):
	final_embedding=np.append(embedding1,embedding2,axis=1)
	return final_embedding,len(final_embedding[0])

def write_file(embedding,data_name,node_number,dim):
	path=data_name + "_onehot_emb%s.dat" % dim
	file=open(path,'w')
	file.writelines([str(node_number),' ',str(dim),'\n'])
	for i in range(node_number):
		file.writelines(str(i))
		for j in range(dim):
			file.writelines([' ',str(embedding[i][j])])
		file.write('\n')
	file.close()

def main():
	
	#data=['leetcode','bilibili1','bilibili2']
	data = ['cora']
	for data_ in data:
		node_num=get_num(data_)
		#embedding=load_embedding(data_,node_num)
		one_hot,dim=load_one_hot(data_,node_num)
		#final_embed,dim=joint(embedding,one_hot)
		print("dim =",dim)
		write_file(one_hot,data_,node_num,dim)



if __name__ == '__main__':
	main()
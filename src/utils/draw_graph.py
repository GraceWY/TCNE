import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import pdb
import random

#color_list=[[247,252,240],[224,243,219],[204,235,197],[168,221,181],\
#[123,204,196],[78,179,211],[43,140,190],[8,104,172],[8,64,129]]

color_list = [[228,26,28],[55,126,184],[77,175,74],[152,78,163],\
[255,127,0],[255,255,51],[166,86,40],[247,129,191],[153,153,153]]

#color_list = [[141,211,199],[255,255,179],[190,186,218],[251,128,114],\
#[128,177,211],[253,180,98],[179,222,105],[252,205,229],[217,217,217]]

"""
            show_nodes=["Binary Search","Stack","String","Array",\
                        "Depth-first Search","Sort","Queue","Heap",\
                        "Topological Sort","Union Find","Tree",\
                        "Greedy","Linked List","Binary Indexed Tree",\
                        "Hash Table","Random","Graph","Reservoir Sampling",\
                        "Breadth-first Search","Trie","Two Pointers"]
"""

'''
show_nodes = ["Tree","Two Pointers","Dynamic Programming",
              "Linked List","Hash Table","Array","Union Find",
              "Binary Search Tree","Greedy","Graph","Breadth-first Search",
              "Depth-first Search","Divide and Conquer"]
'''


show_nodes = ["Tree","Two Pointers","Dynamic Programming",
              "Segment Tree","Queue","Array","Union Find",
              "Binary Search Tree","Greedy","Math","Design",
              "Depth-first Search","Divide and Conquer"]

class DrawGraph(object):
    """This class is for drawing picture in python
    """
    @staticmethod
    def draw_ellipse(mus, std_sigs, row2name, save_path=None, k=1, filter=True):
        N = len(mus)
        """
        {0: 'Random', 1: 'Reservoir Sampling', 2: 'Dynamic Programming', 
        3: 'Greedy', 4: 'Design', 5: 'Queue', 6: 'Hash Table',
         7: 'Linked List', 8: 'Stack', 9: 'Heap', 10: 'Array', 
         11: 'Depth-first Search', 12: 'Tree', 13: 'Binary Search Tree',
          14: 'Map', 15: 'Binary Search', 16: 'Graph', 17: 'String', 
          18: 'Two Pointers', 19: 'Trie', 20: 'Breadth-first Search', 
          21: 'Union Find', 22: 'Math', 23: 'Recursion', 
          24: 'Bit Manipulation', 25: 'Binary Indexed Tree', 
          26: 'Segment Tree', 27: 'Divide and Conquer', 28: 'Sort', 
          29: 'Rejection Sampling', 30: 'Backtracking', 31: 'Brainteaser', 
          32: 'Minimax', 33: 'Topological Sort'}
        """
        if filter==True :

            n=len(show_nodes)
            new_mus=np.zeros((n,2))
            new_sig=np.zeros((n,2))
            new_dic={}
            count=0;
            for i in range(N):
                if row2name[i] in show_nodes:
                    new_dic[count]=row2name[i]
                    new_mus[count,:]=mus[i,:]
                    new_sig[count,:]=std_sigs[i,:]
                    count+=1
          
            mus=new_mus
            std_sigs=new_sig
            row2name=new_dic
            N=n
        
        ws = k*2*std_sigs[:, 0]
        hs = k*2*std_sigs[:, 1]

        # generate ellipse for each dist
        ells = [Ellipse(xy=mus[i], width=ws[i], height=hs[i]) \
                for i in range(N)]

        # draw graph
        #fig, ax = plt.subplots(subplot_kw = {"aspect": "equal"})
        fig, ax = plt.subplots()
        #pdb.set_trace()
        for i in range(N):
            e = ells[i]
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(np.random.rand())
            e.set_alpha(random.randint(3,9)/10)
            e.set_facecolor(np.array(color_list[random.randint(0,8)])/255)
            #ax.annotate(row2name[i], e.get_center(), fontsize = 15)
            ellipse_x, ellipse_y = e.get_center()            
            ellipse_y += std_sigs[i,1]*2+0.3
            #ax.annotate(row2name[i], (ellipse_x,ellipse_y), fontsize = 15)

        
        # set border for the picture
        lft = np.min(mus-k*2*std_sigs)-0.2
        rgt = np.max(mus+k*2*std_sigs)+0.2
        ax.set_xlim(lft, rgt)
        ax.set_ylim(lft, rgt)

        ax.set_xticks([])
        ax.set_yticks([])

        '''
        for tick in ax.get_xticklabels():
            tick.set_fontsize(12)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(12)
        '''

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


    @staticmethod
    def draw_scatter(mus, std_sigs, row2name, save_path=None, k=3, filter = True):
        N = len(mus)
        if filter==True :
            
            n=len(show_nodes)
            new_mus=np.zeros((n,2))
            new_sig=np.zeros((n,2))
            new_dic={}
            count=0;
            for i in range(N):
                if row2name[i] in show_nodes:
                    new_dic[count]=row2name[i]
                    new_mus[count,:]=mus[i,:]
                    new_sig[count,:]=std_sigs[i,:]
                    count+=1
          
            mus=new_mus
            std_sigs=new_sig
            row2name=new_dic
            N=n

        #draw graph
        fig,ax=plt.subplots()
        ax.scatter(mus[:,0], mus[:,1], c='b')
        for i in range(N):
            print("point",i)
            #ax.annotate(row2name[i],(mus[i][0] + 0.2, mus[i][1] + 0.2),fontsize = 15)
        

        lft = np.min(mus-k*2*std_sigs)-0.2
        rgt = np.max(mus+k*2*std_sigs)+0.2
        ax.set_xlim(lft, rgt)
        ax.set_ylim(lft, rgt)

        ax.set_xticks([])
        ax.set_yticks([])

        '''
        for tick in ax.get_xticklabels():
            tick.set_fontsize(12)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(12)
        '''


        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

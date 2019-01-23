import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

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
            show_nodes=["Segment Tree","Backtracking",\
                        "Binary Search","Stack","String","Array",\
                        "Depth-first Search","Sort","Queue","Heap",\
                        "Topological Sort","Union Find","Tree",\
                        "Greedy","Linked List","Binary Indexed Tree",\
                        "Hash Table","Random","Graph","Reservoir Sampling",\
                        "Breadth-first Search","Trie","Math","Two Pointers"]
            n=len(show_nodes)
            new_mus=np.zeros((n,2))
            new_sig=np.zeros((n,2))
            new_dic={}
            count=0;
            for i in range(N):
                if row2name[i]in show_nodes:
                    new_dic[count]=row2name[i]
                    new_mus[count,:]=mus[i,:]
                    new_sig[count,:]=std_sigs[i,:]
                    count+=1
          
            mus=new_mus
            std_sigs=new_sig
            row2name=new_dic
            N=n
            #print("n=",n," count=",count)
            #print(new_dic)
        ws = k*2*std_sigs[:, 0]
        hs = k*2*std_sigs[:, 1]

        # generate ellipse for each dist
        ells = [Ellipse(xy=mus[i], width=ws[i], height=hs[i]) \
                for i in range(N)]

        # draw graph
        fig, ax = plt.subplots(subplot_kw = {"aspect": "equal"})
        for i in range(N):
            e = ells[i]
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(np.random.rand())
            e.set_facecolor(np.random.rand(3))
            ax.annotate(row2name[i], e.get_center(),fontsize=6)
        
        # set border for the picture
        lft = np.min(mus-k*2*std_sigs)-1.0
        rgt = np.max(mus+k*2*std_sigs)+1.0
        ax.set_xlim(lft, rgt)
        ax.set_ylim(lft, rgt)

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


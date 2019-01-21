import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

class DrawGraph(object):
    """This class is for drawing picture in python
    """
    @staticmethod
    def draw_ellipse(mus, std_sigs, row2name, save_path=None, k=1):
        N = len(mus)
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
            ax.annotate(row2name[i], e.get_center())

        # set border for the picture
        lft = np.min(mus-k*2*std_sigs)-1.0
        rgt = np.max(mus+k*2*std_sigs)+1.0
        ax.set_xlim(lft, rgt)
        ax.set_ylim(lft, rgt)

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def graph_info_and_visualizations():
    adj_mat = np.load(r'data/pems_adj_mat.npy')
    graph = nx.from_numpy_array(adj_mat)

    dh = nx.degree_histogram(graph)
    dgs = list(range(len(dh)))
    plt.bar(dgs, dh)
    plt.show()

    plt.loglog(dgs, dh)
    plt.show()
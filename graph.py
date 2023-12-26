import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pandas import value_counts
from torch import Tensor
from pyvis.network import Network
from sklearn.preprocessing import MinMaxScaler

from viz import get_all_y_for_DCRNN


adj_mat = np.load(r"data/pems_adj_mat.npy")
graph = nx.from_numpy_array(adj_mat)


def graph_info_and_visualizations(graph: nx.Graph) -> None:
    dh = nx.degree_histogram(graph)
    dgs = list(range(len(dh)))
    plt.bar(dgs, dh)
    plt.show()
    plt.savefig("degree_histogram.png")
    plt.close()

    plt.loglog(dgs, dh)
    plt.show()
    plt.savefig("loglog_digree_histogram.png")
    plt.show()


def create_weighted_graph(predictions: Tensor, graph: nx.Graph, source: int, target: int):
    node_labels = {
        node: prediction for node, prediction in zip(graph.nodes, predictions)
    }

    edge_weights = {
        (u, v): 1 / prediction if prediction != 0 else 100 for (u, v), prediction in zip(graph.edges, predictions)
    }

    # Set the node labels and edge weights of the graph
    nx.set_node_attributes(graph, node_labels, "label")
    nx.set_edge_attributes(graph, edge_weights, "weight")

    # Compute the shortest path
    shortest_path = nx.dijkstra_path(
        graph, source=source, target=target, weight="weight"
    )

    print(shortest_path)
    
    list_of_edges = []
    for index in range(len(shortest_path)-1):
        list_of_edges.append((shortest_path[index], shortest_path[index+1]))
        
    reversed_list = [(value[1], value[0]) for value in list_of_edges]
    print(list_of_edges)
    print(reversed_list)
        
        

    # Highlight the shortest path
    edge_colors = [
        "red" if edge in list_of_edges or edge in reversed_list else "white"
        for edge in graph.edges
    ]
    node_colors = ["green" if node in shortest_path else "grey" for node in graph.nodes]

    return shortest_path, node_colors, edge_colors


if __name__ == "__main__":
    net = Network(notebook=False, neighborhood_highlight=True)

    timestep = 1
    scaler = MinMaxScaler(feature_range=(0, 100))
    y_pred, y_true = get_all_y_for_DCRNN()
    y_pred = y_pred[timestep, :, 0]
    y_pred = scaler.fit_transform([y_pred])[0]

    shortest_path, node_colors, edge_colors = create_weighted_graph(
        y_pred, graph, source=10, target=300
    )

    for index, (node, color) in enumerate(zip(graph.nodes, node_colors)):
        net.add_node(node, color=color, label=f'Node {index}')

    for edge, color in zip(graph.edges, edge_colors):
        net.add_edge(*edge, color=color)

    net.show("network.html", notebook=False)

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
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


def create_weighted_graph(y_true: Tensor, graph: nx.Graph, source: int, target: int):
    predictions = y_true
    node_labels = {
        node: prediction for node, prediction in zip(graph.nodes, predictions)
    }

    edge_weights = {
        (u, v): 1 / prediction for (u, v), prediction in zip(graph.edges, predictions)
    }

    # Set the node labels and edge weights of the graph
    nx.set_node_attributes(graph, node_labels, "label")
    nx.set_edge_attributes(graph, edge_weights, "weight")

    # Compute the shortest path
    shortest_path = nx.dijkstra_path(
        graph, source=source, target=target, weight="weight"
    )

    print(shortest_path)

    # Highlight the shortest path
    edge_colors = [
        "red" if edge in zip(shortest_path, shortest_path[1:]) else "black"
        for edge in graph.edges
    ]
    node_colors = ["green" if node in shortest_path else "gray" for node in graph.nodes]

    return shortest_path, node_colors, edge_colors


if __name__ == "__main__":
    net = Network(notebook=False)
    
    timestep = 1
    scaler = MinMaxScaler(feature_range=(0, 100))
    y_pred, y_true = get_all_y_for_DCRNN()
    y_pred = y_pred[timestep, :, 0]
    print(y_pred.shape)
    y_pred = scaler.fit_transform([y_pred])

    shortest_path, node_colors, edge_colors = create_weighted_graph(
        y_pred, graph, source=10, target=7
    )

    for node, color in zip(graph.nodes, node_colors):
        net.add_node(node, color=color)

    for edge, color in zip(graph.edges, edge_colors):
        net.add_edge(*edge, color=color)

    net.show("network.html", notebook=False)

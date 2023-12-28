import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pyvis.network import Network
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

from viz import get_all_y_for_DCRNN

SOURCE = 10
TARGET = 232

adj_mat = np.load(r"data/pems_adj_mat.npy")
graph = nx.from_numpy_array(adj_mat)
# print(graph[10][50]['weight'])
# print(adj_mat[50, 10 ])
# print(nx.dijkstra_path(graph,source=SOURCE, target=TARGET))


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


def create_weighted_graph(
    predictions: Tensor, graph: nx.Graph, source: int, target: int, weight_name: str
):
    node_labels = {
        node: prediction for node, prediction in zip(graph.nodes, predictions)
    }

    edge_weights = {
        (u, v): 1 / prediction if prediction != 0 else 100
        for (u, v), prediction in zip(graph.edges, predictions)
    }
    edge_weights_2 = {(u, v): graph[v][u]["weight"] for (u, v) in graph.edges}
    edge_weights_3 = {
        (u, v): edge_weights_2[(u, v)] + edge_weights[(u, v)]
        for (u, v) in edge_weights.keys()
    }

    # Set the node labels and edge weights of the graph
    nx.set_node_attributes(graph, node_labels, "label")
    nx.set_edge_attributes(graph, edge_weights, "weight")
    nx.set_edge_attributes(graph, edge_weights_2, "weight_true")
    nx.set_edge_attributes(graph, edge_weights_3, "combined")

    # Compute the shortest path

    shortest_path = nx.dijkstra_path(
        graph, source=source, target=target, weight=weight_name
    )

    list_of_edges = []
    reversed_list = []

    for index in range(len(shortest_path) - 1):
        list_of_edges.append((shortest_path[index], shortest_path[index + 1]))
        reversed_list.append(list_of_edges[index][::-1])

    # Highlight the shortest path
    edge_colors = [
        "red" if edge in list_of_edges or edge in reversed_list else "black"
        for edge in graph.edges
    ]
    node_colors = ["green" if node in shortest_path else "grey" for node in graph.nodes]

    return shortest_path, node_colors, edge_colors


if __name__ == "__main__":
    timestep = 1
    scaler = MinMaxScaler(feature_range=(0, 100))
    y_pred, y_true = get_all_y_for_DCRNN()
    y_pred = y_pred[timestep, :, 0]
    y_pred = scaler.fit_transform([y_pred])[0]

    for weight_name in ("weight", "weight_true", "combined"):
        net = Network(notebook=False, neighborhood_highlight=True)
        shortest_path, node_colors, edge_colors = create_weighted_graph(
            y_pred, graph, source=SOURCE, target=TARGET, weight_name=weight_name
        )
        print(shortest_path)

        for index, (node, color) in enumerate(zip(graph.nodes, node_colors)):
            net.add_node(node, color=color, label=f"Node {index}")

        for edge, color in zip(graph.edges, edge_colors):
            net.add_edge(*edge, color=color)

        net.show(f"network_{weight_name}.html", notebook=False)

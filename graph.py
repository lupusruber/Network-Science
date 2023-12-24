import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
from pyvis.network import Network

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
    predictions = y_true.cpu().numpy()
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

    # Draw the graph
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors)
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    plt.axis("off")
    plt.show()
    plt.savefig("djikstra.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ones = np.ones(325)
    
    net = Network(notebook=False)

    shortest_path = nx.dijkstra_path(graph, source=10, target=7, weight="weight")
    
    for node in graph.nodes:
        net.add_node(node, color='red' if node in shortest_path else 'blue')

    for edge in graph.edges:
        net.add_edge(*edge, color='red' if set(edge).issubset(shortest_path) else 'blue')
        
    net.show('network.html', notebook=False)

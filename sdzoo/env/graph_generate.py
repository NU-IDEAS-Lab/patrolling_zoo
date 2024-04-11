import networkx as nx
import random
import math
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(pos1, pos2):
    """Compute the Euclidean distance between two points represented as tuples."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def generate_graph_with_coordinates_v3(num_nodes=40, a=40, b=75):
    while True:
        """
        Generate a random graph where the probability of an edge existing between two nodes 
        is inversely proportional to the distance between them.

        Returns:
        - G: A NetworkX graph with nodes having coordinate data.
        """

        # Create an empty graph
        G = nx.Graph()

        # Populate the grid points
        grid_points = [(x, y) for x in range(0, 701, 100) for y in range(0, 501, 100)]

        # Add nodes to graph based on the grid with a 0.2 probability
        node_counter = 0
        for point in grid_points:
            if random.random() < 0.7:
                G.add_node(node_counter, pos=point, visitTime=0.0)
                node_counter += 1

        # Randomly add edges between nodes based on distance
        for u in G.nodes():
            for v in G.nodes():
                if u != v:
                    distance = euclidean_distance(G.nodes[u]['pos'], G.nodes[v]['pos'])
                    probability = 1 / (1 + np.exp((distance - b) / a))
                    if random.random() < probability:
                        G.add_edge(u, v)
        
        if G.number_of_nodes() >= num_nodes:
            break

    return G

def save_graph_to_file(graph, filename):
    with open(filename, 'w') as file:
        # Write graph information
        file.write(f"{graph.number_of_nodes()}\n")
        file.write("1000\n")
        file.write("1000\n")
        file.write("1.0\n")
        file.write("0\n")
        file.write("0\n")
        # Add other graph information here as needed

        # Write node data
        for node, data in graph.nodes(data=True):
            file.write("\n")
            file.write(f"{node}\n")
            x, y = data['pos']
            file.write(f"{x}\n")
            file.write(f"{y}\n")

            # Write edges for the node
            edges = graph.edges(node)
            file.write(f"{len(edges)}\n")
            for edge in edges:
                neighbor = edge[1]
                file.write(f"{neighbor}\n")
                file.write("W\n")
                file.write("10\n")
                # Add other edge information here as needed

# Generate and save the graph
graph = generate_graph_with_coordinates_v3(num_nodes=40, a=20, b=100)
save_graph_to_file(graph, '/home/gyaan/sdzoo/sdzoo/env/graph_file.graph')

# Draw the graph (optional)
pos = nx.get_node_attributes(graph, 'pos')
nx.draw(graph, pos, node_size=20)
plt.show()
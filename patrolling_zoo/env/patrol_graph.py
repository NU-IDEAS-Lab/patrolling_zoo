import networkx as nx
import math
import matplotlib.pyplot as plt

class PatrolGraph():
    ''' This reads a graph file of the format provided by
        https://github.com/davidbsp/patrolling_sim '''

    def __init__(self, filepath):
        self.graph = nx.Graph()
        self.loadFromFile(filepath)

    def loadFromFile(self, filepath: str):
        with open(filepath, "r") as file:
            # Read graph information.
            self.graphDimension = int(file.readline())
            self.widthPixels = int(file.readline())
            self.heightPixels = int(file.readline())
            self.resolution = float(file.readline())
            self.offsetX = float(file.readline())
            self.offsetY = float(file.readline())

            # Read node data.
            for _ in range(self.graphDimension):
                file.readline()
                
                # Create the node.
                i = int(file.readline())
                self.graph.add_node(i,
                    pos = (int(file.readline()) * self.resolution + self.offsetX,
                        int(file.readline()) * self.resolution + self.offsetY)
                )
                
                # Create edges.
                numEdges = int(file.readline())
                for _ in range(numEdges):
                    j = int(file.readline())
                    direction = str(file.readline()) # not useful!
                    cost = int(file.readline())
                    self.graph.add_edge(i, j, weight = cost)

    def getNodePosition(self, node):
        ''' Returns the node position as a tuple (x, y). '''

        return self.graph.nodes[node]["pos"]

    def getOriginsFromInitialPoses(self, initialPoses):
        ''' Given (x,y) initial positions, returns the nearest node for each position.
            This is a horrible n^2 algorithm but that's fine for now. '''

        origins = []
        positions = nx.get_node_attributes(self.graph, 'pos')
        for (x, y) in zip(initialPoses[0::2], initialPoses[1::2]):
            bestDist = math.sqrt(math.pow(positions[0][0] - x, 2) + math.pow(positions[0][1] - y, 2))
            bestNode = 0
            for i in range(len(positions)):
                dist = math.sqrt(math.pow(positions[i][0] - x, 2) + math.pow(positions[i][1] - y, 2))
                if dist < bestDist:
                    bestDist = dist
                    bestNode = i
            origins.append(bestNode)
        return origins
    

    def PlotGraph(self, figsize=(18, 12)):
        """
        PlotGraph function plots the graph using matplotlib and networkx libraries.

        Args:
            figsize (tuple, optional): The size of the figure in inches. Defaults to (18, 12).

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.get_node_attributes(self.graph, 'pos')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=600,font_size=10, font_color='black')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=7)
        # After executing the function do not forget plt.show()

import numpy as np
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
                        int(file.readline()) * self.resolution + self.offsetY),
                    visitTime = 0.0
                )
                
                # Create edges.
                numEdges = int(file.readline())
                for _ in range(numEdges):
                    j = int(file.readline())
                    direction = str(file.readline()) # not useful!
                    cost = int(file.readline()) # we no longer use this cost value, as it does not correspond to the actual euclidean distance.
                    self.graph.add_edge(i, j)
        
        # Set a weight on each edge which corresponds to the actual euclidean distance.
        for edge in self.graph.edges:
            i = edge[0]
            j = edge[1]
            self.graph.edges[i, j]["weight"] = self._dist(self.graph.nodes[i]["pos"], self.graph.nodes[j]["pos"])
        
        # Compute the absolute longest path.
        self.longestPathLength = 0.0
        all_pairs = nx.all_pairs_dijkstra_path_length(self.graph, weight="weight")
        for i in all_pairs:
            if i[1][j] > self.longestPathLength:
                self.longestPathLength = i[1][j]


    def reset(self):
        ''' Resets the graph to initial state. '''

        for node in self.graph.nodes:
            self.graph.nodes[node]["visitTime"] = 0.0


    def getNodePosition(self, node):
        ''' Returns the node position as a tuple (x, y). '''

        return self.graph.nodes[node]["pos"]
    

    def setNodeVisitTime(self, node, timeStamp):
        ''' Sets the node visit time. '''

        self.graph.nodes[node]["visitTime"] = timeStamp
    

    def getNodeVisitTime(self, node):
        ''' Returns the node visit time. '''

        return self.graph.nodes[node]["visitTime"]
    

    def getNodeIdlenessTime(self, node, currentTime):
        ''' Returns the node idleness time. '''

        return currentTime - self.graph.nodes[node]["visitTime"]


    def getAverageIdlenessTime(self, currentTime):
        ''' Returns the average idleness time of all nodes. '''

        nodes = self.graph.nodes
        number_of_nodes = len(nodes)
        return sum([self.getNodeIdlenessTime(node, currentTime) for node in nodes]) / float(number_of_nodes)


    def getWorstIdlenessTime(self, currentTime):
        ''' Returns the worst idleness time of all nodes. '''

        nodes = self.graph.nodes
        return max([self.getNodeIdlenessTime(node, currentTime) for node in nodes])


    def getStdDevIdlenessTime(self, currentTime):
        ''' Returns the standard deviation of idleness time of all nodes. '''

        nodes = self.graph.nodes
        number_of_nodes = len(nodes)
        average_idleness_time = self.getAverageIdlenessTime(currentTime)
        return math.sqrt(sum([math.pow(self.getNodeIdlenessTime(node, currentTime) - average_idleness_time, 2) for node in nodes]) / float(number_of_nodes))


    def getNearestNode(self, pos, epsilon=None):
        ''' Returns the nearest node to the given position.
            If epsilon is not None and no node is within epsilon, returns None. '''
        
        # Find the nearest node.
        bestDist = math.sqrt(math.pow(self.graph.nodes[0]["pos"][0] - pos[0], 2) + math.pow(self.graph.nodes[0]["pos"][1] - pos[1], 2))
        bestNode = 0
        for i in range(len(self.graph.nodes)):
            dist = math.sqrt(math.pow(self.graph.nodes[i]["pos"][0] - pos[0], 2) + math.pow(self.graph.nodes[i]["pos"][1] - pos[1], 2))
            if dist < bestDist:
                bestDist = dist
                bestNode = i

        # Check if the nearest node is within epsilon.
        if epsilon is not None and bestDist > epsilon:
            return None
        else:
            return bestNode


    def getOriginsFromInitialPoses(self, initialPoses):
        ''' Given (x,y) initial positions, returns the nearest node for each position.
            This is a horrible n^2 algorithm but that's fine for now. '''

        origins = []
        for pos in zip(initialPoses[0::2], initialPoses[1::2]):
            origins.append(self.getNearestNode(pos))
        return origins
    
    def _dist(self, pos1, pos2):
        ''' Calculates the Euclidean distance between two points. '''

        return np.sqrt(np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2))
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import random
from enum import IntEnum

class NODE_TYPE(IntEnum):
    OBSERVABLE_NODE = 0
    AGENT = 1
    UNOBSERVABLE_NODE = 2


class PatrolGraph():
    ''' This reads a graph file of the format provided by
        https://github.com/davidbsp/patrolling_sim '''
    
    def __init__(self, filepath = None, numNodes = 40):
        self.graph = nx.Graph()
        if filepath is None:
            self.generateRandomGraph(numNodes)
        else:
            self.loadFromFile(filepath)


    def loadFromFile(self, filepath: str): # TODO: add format for people at each node to be able to test a consistent graph
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
                    visitTime = 0.0,
                    id = i,
                    nodeType = NODE_TYPE.OBSERVABLE_NODE,
                    people = np.random.randint(4), # random integers from [0,4)
                    payloads = 0
                )
                
                # Add a self-loop to the node.
                # self.graph.add_edge(i, i)

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


    def generateRandomGraph(self, numNodes, radius=35, sizeX=200.0, sizeY=200.0, seed=None):
        ''' Generates a random graph with the given parameters. '''

        connected = False
        while not connected:
            pos = {i: (random.uniform(0.0, sizeX), random.uniform(0.0, sizeY)) for i in range(numNodes)}
            self.graph = nx.random_geometric_graph(numNodes, radius, pos=pos, seed=seed)
            connected = nx.is_connected(self.graph)
        
        self.graphDimension = numNodes
        self.widthPixels = sizeX
        self.heightPixels = sizeY
        self.resolution = 1.0
        self.offsetX = 0.0
        self.offsetY = 0.0

        for node in self.graph.nodes:
            self.graph.nodes[node]["visitTime"] = 0.0
            self.graph.nodes[node]["nodeType"] = NODE_TYPE.OBSERVABLE_NODE
            self.graph.nodes[node]["id"] = node
            self.graph.nodes[node]["people"] = np.random.randint(4)
            self.graph.nodes[node]["payloads"] = 0
        for edge in self.graph.edges:
            self.graph.edges[edge]["weight"] = self._dist(self.graph.nodes[edge[0]]["pos"], self.graph.nodes[edge[1]]["pos"])
        self.longestPathLength = 0.0
        all_pairs = nx.all_pairs_dijkstra_path_length(self.graph, weight="weight")
        for i in all_pairs:
            for j in i[1]:
                if i[1][j] > self.longestPathLength:
                    self.longestPathLength = i[1][j]
        
        print(f"Finished generating random graph with {numNodes} nodes and degree {self.graph.degree()}.")


    def reset(self, seed=None, randomizeIds=False, regenerateGraph=False):
        ''' Resets the graph to initial state.
            If regenerateGraph is True, a new random graph is generated. '''

        if regenerateGraph:
            self.generateRandomGraph(self.graphDimension, sizeX=self.widthPixels, sizeY=self.heightPixels, seed=seed)
        
        if randomizeIds:
            # Get random node IDs.
            availableIds = random.sample(range(1000), self.graphDimension)
            for node in self.graph.nodes:
                self.graph.nodes[node]["id"] = availableIds.pop()

        for node in self.graph.nodes:
            self.graph.nodes[node]["visitTime"] = 0.0
            self.graph.nodes[node]["payloads"] = 0


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

    def getNodePayloads(self, node):
        ''' Returns the number of payloads delivered to a node'''

        return self.graph.nodes[node]["payloads"]
    
    def getNodePeople(self, node):
        ''' Returns the number of people at a node'''

        return self.graph.nodes[node]["people"]
    
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


    def getPyTorchGeometricGraph(self):
        ''' Returns a torch_geometric (PyG) graph object. '''

        from torch_geometric.utils.convert import from_networkx
        return from_networkx(self.graph, group_node_attrs=["pos", "visitTime", "people", "payloads"], group_edge_attrs=["weight"])


    def exportToFile(self, filename): # TODO: add format for people at each node to be able to test on a consistent graph
        ''' Exports to a file of the same format as `importFromFile`. '''

        with open(filename, "w") as file:
            # Write graph information.
            file.write(f"{self.graphDimension}\n")
            file.write(f"{self.widthPixels}\n")
            file.write(f"{self.heightPixels}\n")
            file.write(f"{self.resolution}\n")
            file.write(f"{self.offsetX}\n")
            file.write(f"{self.offsetY}\n")

            # Write node data.
            for i in range(self.graphDimension):
                file.write("\n")
                
                # Write the node.
                file.write(f"{i}\n")
                file.write(f"{int((self.graph.nodes[i]['pos'][0] - self.offsetX) / self.resolution)}\n")
                file.write(f"{int((self.graph.nodes[i]['pos'][1] - self.offsetY) / self.resolution)}\n")
                
                # Write edges.
                numEdges = self.graph.degree[i]
                file.write(f"{numEdges}\n")
                for j in self.graph.neighbors(i):
                    file.write(f"{j}\n")
                    file.write("S\n")
                    # Why do we convert to int first? Well, that's what the original patrolling_sim did...
                    file.write(f"{int(round(self.graph.edges[i, j]['weight'] / self.resolution))}\n")
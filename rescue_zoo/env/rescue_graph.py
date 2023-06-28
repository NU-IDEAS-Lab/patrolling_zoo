from .graph_environment import GraphEnvironment

class RescueGraph(GraphEnvironment):
    ''' The GraphEnvironment with additional state information for the rescue scenario. '''

    def reset(self, initialState = {}):
        ''' Resets the graph to initial state. '''

        super().reset()
        for node in self.graph.nodes:
            self.graph.nodes[node]["state"] = initialState
    
    def getNodeState(self, node):
        ''' Returns the node state. '''

        return self.graph.nodes[node]["state"]
    
    def setNodeState(self, node, state):
        ''' Sets the node state. '''

        self.graph.nodes[node]["state"] = state
import unittest
from pettingzoo.test import parallel_api_test
from patrolling_zoo.patrolling_zoo_v0 import PatrolGraph, parallel_env

class TestEnvironment(unittest.TestCase):

    def test_parallel_api(self):
        graph = PatrolGraph("patrolling_zoo/env/cumberland.graph")
        env = parallel_env(graph, num_agents=2)
        parallel_api_test(env, num_cycles=1000)
    
    def test_path_length_from_node0(self):
        graph = PatrolGraph("patrolling_zoo/env/4nodes.graph")
        env = parallel_env(graph, num_agents=1)
        agent = env.agents[0]
        agent.reset()
        agent.position = graph.getNodePosition(0)
        agent.lastNode = 0

        path = env._getPathToNode(agent, 0)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 0.0)

        path = env._getPathToNode(agent, 1)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 40.0)

        path = env._getPathToNode(agent, 2)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 25.0)

        path = env._getPathToNode(agent, 3)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertAlmostEqual(pathLen, 47.2, places=1)

    def test_path_length_from_node1(self):
        graph = PatrolGraph("patrolling_zoo/env/4nodes.graph")
        env = parallel_env(graph, num_agents=1)
        agent = env.agents[0]
        agent.reset()
        agent.position = graph.getNodePosition(1)
        agent.lastNode = 1

        path = env._getPathToNode(agent, 0)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 40.0)

        path = env._getPathToNode(agent, 1)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 0.0)

        path = env._getPathToNode(agent, 2)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 65.0)

        path = env._getPathToNode(agent, 3)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 25.0)
    
    def test_path_length_from_edge13(self):
        graph = PatrolGraph("patrolling_zoo/env/4nodes.graph")
        env = parallel_env(graph, num_agents=1)
        agent = env.agents[0]
        agent.reset()
        pos = graph.getNodePosition(1)
        agent.position = (pos[0], pos[1] + 5.0)
        agent.edge = (1, 3)
        agent.lastNode = 1

        path = env._getPathToNode(agent, 0)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 45.0)

        path = env._getPathToNode(agent, 1)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 5.0)

        path = env._getPathToNode(agent, 2)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 60.0)

        path = env._getPathToNode(agent, 3)
        pathLen = env._getAgentPathLength(agent, path)
        self.assertEqual(pathLen, 20.0)
    
    def test_state(self):
        graph = PatrolGraph("patrolling_zoo/env/4nodes.graph")
        env = parallel_env(graph,
            num_agents = 1,
            observe_method = "ajg_new"
        )
        agent = env.agents[0]
        env.reset(seed=42)
        pos = graph.getNodePosition(1)
        agent.position = (pos[0], pos[1] + 5.0)
        agent.edge = (1, 3)
        agent.lastNode = 1

        state = env.state()
        self.assertEqual(state["vertex_state"][0], 1.0)
        self.assertEqual(state["vertex_state"][1], 1.0)
        self.assertEqual(state["vertex_state"][2], 1.0)
        self.assertEqual(state["vertex_state"][3], 1.0)
        self.assertAlmostEqual(state["vertex_distances"][agent][0], 0.954, places=3)
        self.assertAlmostEqual(state["vertex_distances"][agent][1], 0.106, places=3)
        self.assertAlmostEqual(state["vertex_distances"][agent][2], 1.272, places=3)
        self.assertAlmostEqual(state["vertex_distances"][agent][3], 0.424, places=3)

        self.assertEqual(len(state), 3)
        self.assertEqual(len(state["vertex_state"]), 4)
        self.assertEqual(len(state["vertex_distances"]), 1)
        self.assertEqual(len(state["vertex_distances"][agent]), 4)
    
    def test_movement_node0_node1(self):
        graph = PatrolGraph("patrolling_zoo/env/4nodes.graph")
        env = parallel_env(graph,
            num_agents = 1,
            observe_method = "ajg_new"
        )
        agent = env.agents[0]
        env.reset(seed=42)
        pos = graph.getNodePosition(0)
        agent.position = pos
        agent.edge = None
        agent.lastNode = 0

        # Move agent one step towards node 1.
        reached, stepSize = env._moveTowardsNode(agent, 1, 1.0)
        self.assertEqual(agent.position, (pos[0] + 1.0, pos[1]))
        self.assertEqual(agent.edge, (0, 1))
        self.assertEqual(agent.lastNode, 0)
        self.assertEqual(stepSize, 0.0)
        self.assertEqual(reached, False)

        # Move agent one step towards node 1.
        reached, stepSize = env._moveTowardsNode(agent, 1, 1.0)
        self.assertEqual(agent.position, (pos[0] + 2.0, pos[1]))
        self.assertEqual(agent.edge, (0, 1))
        self.assertEqual(agent.lastNode, 0)
        self.assertEqual(stepSize, 0.0)
        self.assertEqual(reached, False)

        # Move agent one step towards node 0.
        reached, stepSize = env._moveTowardsNode(agent, 0, 1.0)
        self.assertEqual(agent.position, (pos[0] + 1.0, pos[1]))
        self.assertEqual(agent.edge, (0, 1))
        self.assertEqual(agent.lastNode, 0)
        self.assertEqual(stepSize, 0.0)
        self.assertEqual(reached, False)

        # Move agent one step towards node 1.
        reached, stepSize = env._moveTowardsNode(agent, 1, 1.0)
        self.assertEqual(agent.position, (pos[0] + 2.0, pos[1]))
        self.assertEqual(agent.edge, (0, 1))
        self.assertEqual(agent.lastNode, 0)
        self.assertEqual(stepSize, 0.0)
        self.assertEqual(reached, False)

        # Move agent 40 steps towards node 1.
        reached, stepSize = env._moveTowardsNode(agent, 1, 40.0)
        self.assertEqual(agent.position, graph.getNodePosition(1))
        self.assertEqual(agent.edge, None)
        self.assertEqual(agent.lastNode, 1)
        self.assertEqual(stepSize, 2.0)
        self.assertEqual(reached, True)


if __name__ == '__main__':
    unittest.main()
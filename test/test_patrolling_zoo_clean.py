import unittest
from pettingzoo.test import parallel_api_test
from patrolling_zoo.patrolling_zoo_v0 import PatrolGraph
from patrolling_zoo.env.clean_patrolling_zoo import parallel_env

class TestEnvironment(unittest.TestCase):

    def test_parallel_api(self):
        graph = PatrolGraph("patrolling_zoo/env/cumberland.graph")
        env = parallel_env(graph, num_agents=2)
        parallel_api_test(env, num_cycles=1000)

if __name__ == '__main__':
    unittest.main()
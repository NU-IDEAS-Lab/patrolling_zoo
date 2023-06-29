from ..patrolling_zoo_v0 import PatrolGraph, parallel_env

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    graph = PatrolGraph("patrolling_zoo/env/cumberland.graph")
    env = parallel_env(graph, num_agents=2)
    parallel_api_test(env, num_cycles=1000)
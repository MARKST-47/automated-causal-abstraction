import sys
import threading
from contextlib import contextmanager
import _thread
import time
import networkx as nx
from abc import ABC, abstractmethod
import networkit as nk
import copy
from utils import filter_by_max_length

class TimeoutException(Exception):
        pass

class CliqueAnalysers(ABC):
    def __init__(self):
        self.cliques = []
    
    @abstractmethod
    def get_max_cliques(self, G):
        pass

# Bron and Kerbosch algorithm for finding all maximal cliques, iterative version
class ExhaustiveCliqueFinder(CliqueAnalysers):
    def __init__(self):
        super().__init__()
    
    def get_max_cliques(self, G):
        return filter_by_max_length(list(nx.find_cliques(G)))

class DegreeHeuristic(CliqueAnalysers):
    def __init__(self):
        super().__init__()

    def get_max_cliques(self, G):
        degree_sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)

        for node, _ in degree_sorted_nodes:
            clique = [node]
            for neighbor in G.neighbors(node):
                if all(neighbor in G.neighbors(c) for c in clique):
                    clique.append(neighbor)
            self.cliques.append(clique)

        return self.cliques
    
class RemovalHeuristic(CliqueAnalysers):
    def __init__(self):
        super().__init__()

    def get_max_cliques(self, G):
        G_copy = copy.deepcopy(G)
        p = G_copy.number_of_nodes()
        threshold = 5
        
        while threshold <= p:
            max_clique = nx.approximation.max_clique(G_copy)
            p = len(max_clique)
            self.cliques.append(list(max_clique))
            G_copy.remove_nodes_from(max_clique)
        
        return self.cliques

# https://github.com/donfaq/max_clique/blob/master/main.py#L94
class BranchAndBoundHeuristic(CliqueAnalysers):
    def __init__(self, time):
        super().__init__()
        self.time = time
    
    @contextmanager
    def time_limit(self, seconds):
        timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutException()
        finally:
            timer.cancel()


    def timing(f):
        '''
        Measures time of function execution
        '''
        def wrap(*args):
            time1 = time.time()
            ret = f(*args)
            time2 = time.time()
            print('\n{0} function took {1:.3f} ms'.format(
                f.__name__, (time2 - time1) * 1000.0))
            return (ret, '{0:.3f} ms'.format((time2 - time1) * 1000.0))
        return wrap

    @timing
    def get_max_clique(self, graph):
        return self.bb_maximum_clique(graph)

    def bb_maximum_clique(self, graph):
        max_clique = self.greedy_clique_heuristic(graph)
        chromatic_number = self.greedy_coloring_heuristic(graph)
        if len(max_clique) == chromatic_number:
            return max_clique
        else:
            g1, g2 = self.branching(graph, len(max_clique))
            return max(self.bb_maximum_clique(g1), self.bb_maximum_clique(g2), key=lambda x: len(x))

    def branching(self, graph, cur_max_clique_len):
        '''
        Branching procedure
        '''
        g1, g2 = graph.copy(), graph.copy()
        max_node_degree = len(graph) - 1
        nodes_by_degree = [node for node in sorted(nx.degree(graph),  # All graph nodes sorted by degree (node, degree)
                                                key=lambda x: x[1], reverse=True)]
        # Nodes with (current clique size < degree < max possible degree)
        partial_connected_nodes = list(filter(
            lambda x: x[1] != max_node_degree and x[1] <= max_node_degree, nodes_by_degree))
        # graph without partial connected node with highest degree
        g1.remove_node(partial_connected_nodes[0][0])
        # graph without nodes which is not connected with partial connected node with highest degree
        g2.remove_nodes_from(
            graph.nodes() -
            graph.neighbors(
                partial_connected_nodes[0][0]) - {partial_connected_nodes[0][0]}
        )
        return g1, g2

    def greedy_coloring_heuristic(self, graph):
        '''
        Greedy graph coloring heuristic with degree order rule
        '''
        color_num = iter(range(0, len(graph)))
        color_map = {}
        used_colors = set()
        nodes = [node[0] for node in sorted(nx.degree(graph),
                                            key=lambda x: x[1], reverse=True)]
        color_map[nodes.pop(0)] = next(color_num)  # color node with color code
        used_colors = {i for i in color_map.values()}
        while len(nodes) != 0:
            node = nodes.pop(0)
            neighbors_colors = {color_map[neighbor] for neighbor in
                                list(filter(lambda x: x in color_map, graph.neighbors(node)))}
            if len(neighbors_colors) == len(used_colors):
                color = next(color_num)
                used_colors.add(color)
                color_map[node] = color
            else:
                color_map[node] = next(iter(used_colors - neighbors_colors))
        return len(used_colors)

    def greedy_clique_heuristic(self, graph):
        '''
        Greedy search for clique iterating by nodes 
        with highest degree and filter only neighbors 
        '''
        K = set()
        nodes = [node[0] for node in sorted(nx.degree(graph),
                                            key=lambda x: x[1], reverse=True)]
        while len(nodes) != 0:
            neigh = list(graph.neighbors(nodes[0]))
            K.add(nodes[0])
            nodes.remove(nodes[0])
            nodes = list(filter(lambda x: x in neigh, nodes))
        return K
    
    def get_max_cliques(self, graph):
        try:
            with self.time_limit(self.time):
                max_clq = self.get_max_clique(graph)
                # self.cliques = max_clq
                print('\nMaximum clique', max_clq, '\nlen:', len(max_clq))
                return max_clq
        except TimeoutException:
            print("Timed out!")
            sys.exit(0)

class MaxCliqueHeuristic(CliqueAnalysers):
    def __init__(self):
        super().__init__()
    
    def get_max_cliques(self, G):
        return [list(nx.approximation.max_clique(G))]
    
class MaxCliqueHeuristic_v2(CliqueAnalysers):
    def __init__(self):
        super().__init__()
    
    def get_max_cliques(self, G):
        nk_graph = nk.nxadapter.nx2nk(G)
        clique_finder = nk.clique.MaximalCliques(nk_graph, maximumOnly=True)
        clique_finder.run()
        return clique_finder.getCliques()
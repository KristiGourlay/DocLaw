import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from collections import defaultdict


df = pd.read_csv('../data/citation_df.csv', index_col=0)



class Graphing():

    def __init__(self, case):
        self.case = case

    def subset(self):
        subset = int(df[df.case_citation_name == self.case].lda_sub_group)
        return subset

    def cases_cited(self):

        subset = self.subset()
        new_df = df[df.lda_sub_group == subset]
        new_df = new_df[['case_citation_name', 'citations']]
        new_graph = dict(zip(new_df.case_citation_name, new_df.citations))

        cases_that_cited_it = []
        for key, value in new_graph.items():
            if self.case in value:
                cases_that_cited_it.append(key)

        return cases_that_cited_it


    def generate_edges(self, graph):

        edges = []
        for node in graph:
            try:
                for neighbour in graph[node].split(','):

                    edges.append((node, neighbour[2:-2]))
            except:
                pass

        return edges


    def build_graph(self):

        subset = self.subset()
        new_df = df[df.lda_sub_group == subset]
        new_df = new_df[['case_citation_name', 'citations']]
        new_graph = dict(zip(new_df.case_citation_name, new_df.citations))
        new_edges = self.generate_edges(new_graph)
        new_edges = list(set(new_edges))

        new_G = nx.DiGraph()
        new_G.add_edges_from(new_edges)

        return new_G

    def find_connections(self, targets):

        # subset = self.subset()
        graph = self.build_graph()
        paths = []

        for source in targets:
            for target in targets:
                for path in nx.all_simple_paths(graph, source=source, target=target, cutoff=3):
                    paths.extend(path)

        return list(set(paths))


results = ['412 Ill. 285', '326 Ill. 405', '27 Ill. 2d 609', '59 Ill. 2d 502', '96 Ill. App. 3d 1108', '341 Ill. 138', '9 Ill. App. 3d 100', '104 Ill. 2d 302', '27 Ill. 2d 609', '78 Ill. App. 3d 627', '108 Ill. 2d 286', '114 Ill. 2d 209', '82 Ill. 2d 31', '71 Ill. 2d 563', '103 Ill. 2d 266']
example = '27 Ill. 2d 609'

gt = Graphing(example)
gt.subset()
gt.cases_cited()

gt.build_graph()
gt.find_connections(targets=results)

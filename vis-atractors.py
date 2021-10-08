import networkx as nx
# import matplotlib.pyplot as plt
from pyvis.network import Network
from dimcli.utils.networkviz import NetworkViz
from math import log10
import argparse
import json

def getAtractorLabel(atractor, nodes):
    label = []
    for i in range(len(nodes)):
        for s in atractor:
            if (s>>i)%2 == 1:
                label.append(nodes[i])

    return ', '.join(label)

def viz(atractors):
    atractorsf = open(atractors, 'r')
    atractor_data = json.load(atractorsf)
    G = nx.Graph()
    for i in range(len(atractor_data['atractors'])):
        label = getAtractorLabel(atractor_data['atractors'][i], atractor_data['nodes'])
        print(label)
        nx.add_cycle(G, atractor_data['atractors'][i], title=label)
        
    nt = NetworkViz(notebook=True)
    nt.from_nx(G)
    nt.show('nx.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualização de atratores com NetworkX.')
    # parser.add_argument('equation', type=str, help='Arquivo com equações booleans da rede.')
    parser.add_argument('atractors', type=str, help='Arquivo .json com atratores da rede.')
    args = parser.parse_args()
    viz(args.atractors)

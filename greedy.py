# coding: utf-8


import networkx as nx
import numpy as np

import argparse
argumentparser = argparse.ArgumentParser()
argumentparser.add_argument('nnodes', type=int)
argumentparser.add_argument('degree', type=int)

def greedy(g):
    improved = False
    aspl = nx.average_shortest_path_length(g)
    edges = g.edges()
    n_edges = len(edges)
    while not improved:
        id1, id2 = np.random.choice(n_edges, 2, replace=False)
        e1, e2 = edges[id1], edges[id2]
        which = np.random.randint(2)
        new_e1 = (e1[0], e2[which])
        new_e2 = (e1[1], e2[1-which])
        #print(e1, e2, new_e1, new_e2)
        if g.has_edge(*new_e1) or g.has_edge(*new_e2):
            #print('has edge')
            continue
        else:
            g.remove_edges_from([e1, e2])
            g.add_edges_from([new_e1, new_e2])
            if nx.is_connected(g):
                new_aspl = nx.average_shortest_path_length(g)
                if new_aspl < aspl:
                    g.remove_edges_from([new_e1, new_e2])
                    g.add_edges_from([e1, e2])
                    return e1, e2, new_e1, new_e2, which
                else:
                    pass
                    #print('higher aspl')
            else:
                pass
                #print('unconnected')
            g.remove_edges_from([new_e1, new_e2])
            g.add_edges_from([e1, e2])

    
def main(args):
    nnodes = args.nnodes
    degree = args.degree
    assert degree < nnodes
    for _ in range(100):
        g = nx.random_regular_graph(degree, nnodes)
        if nx.is_connected(g):
            print('c')
            print(nx.average_shortest_path_length(g))
            result = greedy(g)
            print(nx.average_shortest_path_length(g))
        else:
            print('u')
            
def save_edges(g, filepath):
	nx.write_edgelist(g, filepath, data=False)
	return

def save_image(g, filepath):
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	
# 	layout = nx.spring_layout(g)
	layout = nx.circular_layout(g)
	nx.draw(g, with_labels=False, node_size=50, linewidths=0, alpha=0.5, node_color='#3399ff', edge_color='#666666', pos=layout)
	plt.draw()
	plt.savefig(filepath)
	return

def save_json(author, email, text1, graph_file, filepath):
	import os
	import json
	from datetime import datetime
	
	metadata = {}
	metadata['author'] = author
	metadata['email'] = email
	metadata['text1'] = text1
	metadata['disclose'] = True
	metadata['graph_file'] = os.path.basename(graph_file)
	metadata['stamp'] = datetime.isoformat(datetime.utcnow())
	with open(filepath, 'w') as f:
		json.dump(metadata, f, indent=1)
	return

def lower_bound_of_diam_aspl(nnodes, degree):
	diam = -1
	aspl = 0.0
	n = 1
	r = 1
	while True:
		tmp = n + degree * pow(degree - 1, r - 1)
		if tmp >= nnodes:
			break
		n = tmp
		aspl += r * degree * pow(degree - 1, r - 1)
		diam = r
		r += 1
	diam += 1
	aspl += diam * (nnodes - n)
	aspl /= (nnodes - 1)
	return diam, aspl

def max_avg_for_matrix(data):
	cnt = 0
	sum = 0.0
	max = 0.0
	for i in data:
		for j in data[i]:
			if i != j:
				cnt += 1
				sum += data[i][j]
				if max < data[i][j]:
					max = data[i][j]
	return max, sum / cnt

if __name__ == '__main__':
    main(argumentparser.parse_args())

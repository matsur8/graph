# -*- coding:utf-8 -*-

import networkx as nx
import numpy as np
import tensorflow as tf

import greedy

np.random.seed(1)
tf.set_random_seed(2)

G = [[0,1,1,0,0],
     [1,0,0,1,0],
     [1,0,0,0,1],
     [0,1,0,0,1],
     [0,0,1,1,0]]

N = 32
d = 3
m = 64
ids = tf.placeholder(dtype=tf.int32, shape=(N,d))
true_aspl = tf.placeholder(dtype=tf.float32)

def graph_conv(inputs, ids, output_dim, non_linear='relu'):
    from_nb_pre = tf.reduce_mean(tf.nn.embedding_lookup(inputs, ids), 1)
    from_nb = tf.layers.dense(from_nb_pre, output_dim)
    from_self = tf.layers.dense(inputs, output_dim, use_bias=False)
    if non_linear == 'relu':
        return tf.nn.relu(from_nb + from_self)
    elif non_linear is None:
        return from_nb + from_self

#inputs = tf.constant(np.random.normal(size=(N,m), scale=1/np.sqrt(m)).astype(np.float32))
inputs = tf.random_normal(shape=(N,m), stddev=1/np.sqrt(m))
h1 = graph_conv(inputs, ids, m)
h2 = graph_conv(h1, ids, m)
h3 = graph_conv(h2, ids, m)
h4 = graph_conv(h3, ids, m, non_linear=None)
relu_h4 = tf.nn.relu(h4)
max_pool = tf.reduce_max(relu_h4, 0, keep_dims=True)
#h2_1 = tf.layers.dense(max_pool, m, activation=tf.nn.relu)  
#h2_2 = tf.reshape(tf.layers.dense(h2_1, 4*m), (m,4))
#score = tf.transpose(tf.matmul(h4, h2_2))
#cross_ent = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=true_action,
#                                                                  logits=score))
#global_h = tf.reshape(relu_h4, (1,N*m))
h3_1 = tf.layers.dense(max_pool, m, activation=tf.nn.relu)  
h3_2 = tf.layers.dense(h3_1, m, activation=tf.nn.relu)  
y = tf.reduce_mean(tf.matmul(h4, tf.transpose(h3_2)))
mse = (y - true_aspl)**2

optimizer = tf.train.AdamOptimizer(1e-4)
#train_step = optimizer.minimize(cross_ent + mse)
#train_step = optimizer.minimize(cross_ent)
train_step = optimizer.minimize(mse)

#action = tf.argmax(score, axis=0)
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

con = False


training_set = [nx.random_regular_graph(d, N) for _ in range(10000)]
validation_set = [nx.random_regular_graph(d, N) for _ in range(1000)]
training_set = [g for g in training_set if nx.is_connected(g)]
validation_set = [g for g in validation_set if nx.is_connected(g)]
#training_set[-1] = validation_set[-1]
n_val = len(validation_set)
print(n_val)
def normalize_edges(g):
    return g.edge
    return tuple(sorted([sorted(e) for e in g.edges()]))
training_edge = [g.edge for g in training_set]
validation_set = [g for g in validation_set if g.edge not in training_edge]
n_train = len(training_set)
n_val = len(validation_set)
print(n_val)
#validation_set = [validation_set[0] for _ in validation_set]
train_aspl = [nx.average_shortest_path_length(g) for g in training_set]
val_aspl = [nx.average_shortest_path_length(g) for g in validation_set]
print('variance train_aspl', np.var(train_aspl))
print('variance valid_aspl', np.var(val_aspl))

for t in range(5000):
    for g, aspl in zip(training_set, train_aspl):
        edges = np.array([x[1] for x in sorted([(n, sorted(list(elist.keys()))) for n, elist in g.edge.items()])], dtype=np.int32)
        sess.run(train_step, feed_dict={ids:edges, true_aspl:aspl})
    mse_train = 0
    for g, aspl in zip(training_set[:1000], train_aspl[:1000]):
        edges = np.array([x[1] for x in sorted([(n, sorted(list(elist.keys()))) for n, elist in g.edge.items()])], dtype=np.int32)
        mse_train += sess.run(mse, feed_dict={ids:edges, true_aspl:aspl})
    print(t, 'train', mse_train / (1.0 * 1000))
    mse_val = 0
    for g, aspl in zip(validation_set, val_aspl):
        edges = np.array([x[1] for x in sorted([(n, sorted(list(elist.keys()))) for n, elist in g.edge.items()])],
                         dtype=np.int32)
        mse_val += sess.run(mse, feed_dict={ids:edges, true_aspl:aspl})
    print(t, 'valid', mse_val / (1.0 * n_val))
    y_val = []
    g = validation_set[0]
    aspl = val_aspl[0]
    edges = np.array([x[1] for x in sorted([(n, sorted(list(elist.keys()))) for n, elist in g.edge.items()])],
                     dtype=np.int32)
    for _ in range(100):
        y_tmp = sess.run(y, feed_dict={ids:edges, true_aspl:aspl})
        y_val.append(y_tmp)
    print(t, 'variance', np.var(y_val))
    
            
    

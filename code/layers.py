import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *


class GraphConvolution():
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            x = inputs
            x = tf.compat.v1.nn.dropout(x, 1-self.dropout)
            x = tf.compat.v1.matmul(x, self.vars['weights'])
            x = tf.compat.v1.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs

class GraphConvolutionSparse():
    def __init__(self, input_dim, output_dim, adj, embeddings_nonzero, name, dropout=0., act=tf.compat.v1.nn.relu):
        self.name = name
        self.vars = {}
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.embeddings_nonzero = embeddings_nonzero

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.embeddings_nonzero)
            x = tf.compat.v1.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.compat.v1.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs

class ETG():
    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.variable_scope(self.name + '_w'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='w')
        self.layer1 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.layer3 = tf.keras.layers.Dense(units=input_dim, activation=tf.nn.relu)

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R_branch_1 = tf.matmul(R, self.vars['weights'])
            R_temp = self.layer1(R)
            R_temp = self.layer2(R_temp)
            R_branch_2 = self.layer3(R_temp)
            R = 0.5 * (R_branch_1 + R_branch_2)
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs
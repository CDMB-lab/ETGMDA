import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from layers import GraphConvolution, GraphConvolutionSparse, ETG
from utils import *


class GCNModel():
    def __init__(self, placeholders, num_embeddings, emb_dim, embeddings_nonzero, adj_nonzero, num_r, name, act=tf.nn.elu):
        self.name = name
        self.inputs = placeholders['embeddings']
        self.input_dim = num_embeddings
        self.emb_dim = emb_dim
        self.embeddings_nonzero = embeddings_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.att = tf.compat.v1.Variable(tf.constant([0.5, 0.33, 0.25]))
        self.num_r = num_r
        with tf.compat.v1.variable_scope(self.name):
            self.build()

    def build(self):
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)
        self.hidden1 = GraphConvolutionSparse(
            name='layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            embeddings_nonzero=self.embeddings_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden2 = GraphConvolution(
            name='layer1',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden1)

        self.emb = GraphConvolution(
            name='layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)

        self.embeddings = self.hidden1 * \
            self.att[0]+self.hidden2*self.att[1]+self.emb*self.att[2]

        self.reconstructions = ETG(
            name='etg',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.nn.sigmoid
            )(self.embeddings)
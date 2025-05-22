from ogb.nodeproppred import PygNodePropPredDataset
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import argparse
import os
import os.path as osp
import csv
import matplotlib.pyplot as plt
import faiss
import time
import warnings
import copy
import json
import pickle
import networkx as nx
import math

from ast import arg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from torch_geometric.datasets import CoraFull, Reddit2, Flickr, Planetoid, Reddit, Amazon
from torch_geometric.utils import add_remaining_self_loops, to_undirected, subgraph, get_laplacian, convert
from torch_geometric.utils.loop import remove_self_loops 
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, SGConv, APPNP, GATConv
import torch_geometric.transforms as T
from torch_geometric.utils.augmentation import mask_feature, add_random_edge
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.transforms import KNNGraph
from torch_geometric.nn.pool.approx_knn import approx_knn_graph
from torch_geometric.nn.pool.knn import MIPSKNNIndex #ApproxMIPSKNNIndex
from torch_geometric.nn import LabelPropagation
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling


from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
from torch import Tensor
import torch_sparse
import scipy.sparse as sp
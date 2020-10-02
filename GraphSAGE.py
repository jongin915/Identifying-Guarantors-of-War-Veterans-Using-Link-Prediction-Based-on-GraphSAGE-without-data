'''
CPU(intel xeon cpu e5-2673 v3 @ 2.40ghz), 8GB RAM, Windows OS
Python==3.6.8, Stellargraph==1.2.1, Tensorflow==2.1.0, linkpred==0.5.1.
'''

import stellargraph as sg
import networkx as nx
import pandas as pd
import numpy as np
import scipy
import itertools
import os

import matplotlib.pyplot as plt

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.calibration import expected_calibration_error, plot_reliability_diagram
from stellargraph.calibration import IsotonicCalibration, TemperatureCalibration

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import accuracy_score

from stellargraph import globalvar
from stellargraph import datasets
import tensorflow as tf

# create whole network
######################
edges = pd.DataFrame({
    "source": Source,
    "target": Target,
    "weight": Value
})
Gs = sg.StellarGraph(nodes=nodes,edges=edges)

# Define an edge splitter on the original graph G:
edge_splitter_train = EdgeSplitter(Gs)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
  p=0.1, method="global", keep_connected=True)

train_edges = G_train.edge_arrays(include_edge_weight=True)

''' extract train set and test set for baseline model
train1 = train_edges[0]
train2 = train_edges[1]
train3 = train_edges[3]

train_data = {'source' : train1, "target" : train2, "weight" : train3}

df = pd.DataFrame(train_data)

df.to_csv("train.csv", mode='w')
'''

batch_size = 256 #para
epochs = 100
num_samples = [10, 10] #para
layer_sizes = [20, 20]

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples, weighted=True)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

# we use MaxPoolingAggregator, MeanPoolingAggregator and MeanAggregator
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, aggregator = sg.layer.MaxPoolingAggregator, generator=train_gen, bias=True, dropout=0.5)

x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method="ip")(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.binary_crossentropy,
    metrics=[tf.keras.metrics.Precision(name='precision')\
                          ,tf.keras.metrics.Recall(name='recall')])

init_train_metrics = model.evaluate(train_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
  print("\t{}: {:0.4f}".format(name, val))

history = model.fit(train_flow, epochs=epochs, verbose=2) #, callbacks=[early_stopping]

train_metrics = model.evaluate(train_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

train_f1 = 1/(1/train_metrics[1]+1/train_metrics[2])*2
print("train_f1 : ",train_f1)

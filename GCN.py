'''
CPU(intel xeon cpu e5-2673 v3 @ 2.40ghz), 8GB RAM, Windows OS
Python==3.6.8, Stellargraph==1.2.1, Tensorflow==2.1.0, linkpred==0.5.1.
'''

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding
import pandas as pd
from tensorflow import keras
from tensorflow.keras import backend as K

# define recall, precision, f1 score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# create whole network
######################
edges = pd.DataFrame({
    "source": Source,
    "target": Target,
    "weight": Value
})
Gs = sg.StellarGraph(nodes=nodes,edges=edges)

import tensorflow as tf

# Define an edge splitter on the original graph G:
edge_splitter_train = EdgeSplitter(Gs)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
  p=0.1, method="global", keep_connected=True)

epochs = 300

train_gen = FullBatchLinkGenerator(G_train, method="gcn", weighted=True)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

layer_sizes = [20, 20]

gcn = GCN(
        layer_sizes=layer_sizes,
        activations=["elu","softmax"],
        generator=train_gen,
        dropout=0.5)

x_inp, x_out = gcn.in_out_tensors()

prediction = LinkEmbedding(activation="relu", method="ip")(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

# use adam optimizers and set learning rate
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc", f1_m, precision_m, recall_m])

init_train_metrics = model.evaluate(train_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
  print("\t{}: {:0.4f}".format(name, val))

history = model.fit(train_flow, epochs=epochs, verbose=2, shuffle=False) #, callbacks=[early_stopping]

# evaluate model
train_metrics = model.evaluate(train_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

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
index = pd.read_csv("Nodelist.csv")
p = index.node.tolist()
node = []
for i in p:
    node.append(str(i))

index2 = pd.read_csv("Nodefeatures.csv")

f1 = index2.division_cap.tolist()
features1 = []
for i in f1:
    features1.append(int(i))

f2 = index2.division_1.tolist()
features2 = []
for i in f2:
    features2.append(int(i))

f3 = index2.division_3.tolist()
features3 = []
for i in f3:
    features3.append(int(i))

f4 = index2.division_6.tolist()
features4 = []
for i in f4:
    features4.append(int(i))

f5 = index2.division_7.tolist()
features5 = []
for i in f5:
    features5.append(int(i))

f6 = index2.division_8.tolist()
features6 = []
for i in f6:
    features6.append(int(i))

f7 = index2.regiment_22.tolist()
features7 = []
for i in f7:
    features7.append(int(i))

f8 = index2.regiment_23.tolist()
features8 = []
for i in f8:
    features8.append(int(i))

f9 = index2.regiment_26.tolist()
features9 = []
for i in f9:
    features9.append(int(i))

f10 = index2.regiment_99.tolist()
features10 = []
for i in f10:
    features10.append(int(i))

f11 = index2.regiment_91.tolist()
features11 = []
for i in f11:
    features11.append(int(i))

f12 = index2.regiment_18.tolist()
features12 = []
for i in f12:
    features12.append(int(i))

f13 = index2.regiment_11.tolist()
features13 = []
for i in f13:
    features13.append(int(i))

f14 = index2.regiment_12.tolist()
features14 = []
for i in f14:
    features14.append(int(i))

f15 = index2.regiment_15.tolist()
features15 = []
for i in f15:
    features15.append(int(i))

f16 = index2.regiment_92.tolist()
features16 = []
for i in f16:
    features16.append(int(i))

f17 = index2.regiment_97.tolist()
features17 = []
for i in f17:
    features17.append(int(i))

f18 = index2.regiment_19.tolist()
features18 = []
for i in f18:
    features18.append(int(i))

f19 = index2.regiment_10.tolist()
features19 = []
for i in f19:
    features19.append(int(i))

f20 = index2.regiment_16.tolist()
features20 = []
for i in f20:
    features20.append(int(i))

f21 = index2.regiment_21.tolist()
features21 = []
for i in f21:
    features21.append(int(i))

f22 = index2.regiment_93.tolist()
features22 = []
for i in f22:
    features22.append(int(i))

f23 = index2.regiment_95.tolist()
features23 = []
for i in f23:
    features23.append(int(i))

f24 = index2.regiment_98.tolist()
features24 = []
for i in f24:
    features24.append(int(i))

f25 = index2.period1.tolist()
features25 = []
for i in f25:
    features25.append(int(i))

f26 = index2.period2.tolist()
features26 = []
for i in f26:
    features26.append(int(i))

f27 = index2.period3.tolist()
features27 = []
for i in f27:
    features27.append(int(i))

f28 = index2.period4.tolist()
features28 = []
for i in f28:
    features28.append(int(i))

f29 = index2.KIA.tolist()
features29 = []
for i in f29:
    features29.append(int(i))

nodes = pd.DataFrame(
    {"a": features1, "b": features2, "c": features3, "d": features4, "e": features5, "f": features6, "g": features7,
     "h": features8, "i": features9,
     "j": features10, "k": features11, "l": features12, "m": features13, "n": features14, "o": features15,
     "p": features16, "q": features17, "r": features18,
     "s": features19, "t": features20, "u": features21, "v": features22, "w": features23, "x": features24,
     "y": features25, "z": features26, "aa": features27,
     "ab": features28, "ac": features29}, index=node)

pos = pd.read_csv("Edgelist.csv")
k = pos.Source.tolist()
l = pos.Target.tolist()
m = pos.Value.tolist()

Source = []
Target = []
Value = []

for i in k:
  Source.append(str(i))

for j in l:
  Target.append(str(j))

for t in m:
  Value.append(int(t))

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

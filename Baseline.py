'''
CPU(intel xeon cpu e5-2673 v3 @ 2.40ghz), 8GB RAM, Windows OS
Python==3.6.8, Stellargraph==1.2.1, Tensorflow==2.1.0, linkpred==0.5.1.
'''

import linkpred

G_train = linkpred.read_network("train1.net") # train dataset
G_entire = linkpred.read_network("whole.net") # entire dataset

test = G_entire
training = G_train
test.remove_edges_from(training.edges()) # create testset

###### 1. Jaccard
jaccard = linkpred.predictors.Jaccard(training, excluded=training.edges()) # train model based on jaccard index
jaccard_results = jaccard.predict() # predict testset based on jaccard index
test_set = set(linkpred.evaluation.Pair(u,v) for u,v in test.edges())
evaluation1 = linkpred.evaluation.EvaluationSheet(jaccard_results, test_set) # compare predicted value with real value
p1 = evaluation1.precision()
f1 = evaluation1.f_score()
r1 = evaluation1.recall()

print("Jaccard p : ",p1[np.argmax(f1)])
print("Jaccard r : ",r1[np.argmax(f1)]) # extract precision and recall when f1-score is optimal
print("Jaccard f1 : ",max(f1)) # extract optimal f1-score

###### 2. CommonNeighbours
CN = linkpred.predictors.CommonNeighbours(training, excluded=training.edges()) # train model based on Common Neighbours
CN_results = CN.predict() # predict testset based on Common Neighbours
test_set = set(linkpred.evaluation.Pair(u,v) for u,v in test.edges()) # compare predicted value with real value
evaluation2 = linkpred.evaluation.EvaluationSheet(CN_results, test_set)
p2 = evaluation2.precision()
f2 = evaluation2.f_score()
r2 = evaluation2.recall()

print("CommonNeighbours p : ",p2[np.argmax(f2)])
print("CommonNeighbours r : ",r2[np.argmax(f2)]) # extract precision and recall when f1-score is optimal
print("CommonNeighbours f1 : ",max(f2)) # extract optimal f1-score

###### 3. ResourceAllocation
RA = linkpred.predictors.ResourceAllocation(training, excluded=training.edges()) # train model based on Resource Allocation
RA_results = RA.predict() # predict testset based on Resource Allocation
test_set = set(linkpred.evaluation.Pair(u,v) for u,v in test.edges()) # compare predicted value with real value
evaluation3 = linkpred.evaluation.EvaluationSheet(RA_results, test_set)
p3 = evaluation3.precision()
f3 = evaluation3.f_score() 
r3 = evaluation3.recall() 

print("ResourceAllocation p : ",p3[np.argmax(f3)])
print("ResourceAllocation r : ",r3[np.argmax(f3)]) # extract precision and recall when f1-score is optimal
print("ResourceAllocation f1 : ",max(f3)) # extract optimal f1-score


###### 4. AdamicAdar
adamicadar = linkpred.predictors.AdamicAdar(training, excluded=training.edges()) # train model based on Adamic Adar
adamicadar_results = adamicadar.predict() # predict testset based on Adamic Adar
test_set = set(linkpred.evaluation.Pair(u,v) for u,v in test.edges()) # compare predicted value with real value
evaluation4 = linkpred.evaluation.EvaluationSheet(adamicadar_results, test_set)
p4 = evaluation4.precision()
f4 = evaluation4.f_score()
r4 = evaluation4.recall()

print("AdamicAdar p : ",p4[np.argmax(f4)])
print("AdamicAdar r : ",r4[np.argmax(f4)]) # extract precision and recall when f1-score is optimal
print("AdamicAdar f1 : ",max(f4)) # extract optimal f1-score
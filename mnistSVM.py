'''
Python code for learning handwritten digits from MNIST data set using SVM
'''
import numpy as np
from sklearn.svm import SVC
import imagetools as imt
import csv
import pickle

# some variables
trfile='data/train.csv'
testfile='data/test.csv'
outfile = 'predictions/pred_SVM.csv'
splitsize=0.6
paramc = [0.1, 0.5, 1, 5, 10 ]
paramg = [0, 0.01, 0.05, 0.1, 0.5, 1]

# load labels
labels = imt.readlabels(trfile)
m = len(labels)

# split to train and cv sets
trsplit = int(np.round(m*splitsize))
iterm=range(m)

nparams = len(paramc)*len(paramg)
scores = np.zeros(nparams)

# load in training set
# note: this takes up a bit of memory
print('Loading training data...')
trsetfull = imt.loadimgfromcsv(trfile,colstart=1)/255.0
print('...done')
trset = trsetfull[:trsplit]
cvset = trsetfull[trsplit:]
trlabels = labels[:trsplit]
cvlabels = labels[trsplit:]
c=5.0
g=0.05
#counter=0
#paramholder=np.zeros([nparams,2])
#bestscore=-1
#for c in paramc:
#	for g in paramg:
#		model=SVC(C=c,gamma=g)
#		print('Fitting model with params c: %f, g: %f ...'%(c,g))
#		model=model.fit(trset,trlabels)
#		try:
#			scorei=model.score(cvset,cvlabels)
#		except:
#			scorei=0 # something went wrong...
#		scores[counter]=np.mean(scorei)
#		paramholder[counter,0]=c
#		paramholder[counter,1]=g
#		if scorei>bestscore:
#			bestscore=scorei
#			bestmodel=model
#		counter+=1
#		print('Score = %f with c: %f, g: %f' %(scorei,c,g))
#bestc=paramholder[scores.argmax(),0]
#bestg=paramholder[scores.argmax(),1]
#print('Best score of %f with c: %f, g: %f' %(bestscore,bestc,bestg))
print('Fitting model with params c: %f, g: %f ...'%(c,g))
model=SVC(C=c,gamma=g)
model=model.fit(trset,trlabels)

pickle.dump(model,open('data/svm_c5_g05.p','wb'))

# predict test set
testx=imt.loadimgfromcsv(testfile,colstart=0)/255.0
ntest=testx.shape[0]
preds = model.predict(testx)
# output to .csv file labels
imt.writeoutpred(map(int,preds),np.arange(ntest)+1,outfile,header=["ImageId","Label"])
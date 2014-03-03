'''
Python code for learning handwritten digits from MNIST data set
'''
import imagetools as imt
import NNtools as nnt
import numpy as np
import time
import csv
import pickle

debug=False

# some variables
trfile='data/train.csv'
testfile='data/test.csv'
outfile = 'predictions/pred.csv'
acttmpfileroot='data/tmp'
# epochs, layers, split size
epochs=15
insz=28*28
outsz=10
hlayers=[200,200,200]
noisel=[0.25,0.25,0.25]
etype='CE'
lr=0.001
splitsize = 0.8 # what portion goes to train vs cross validation
blocksize=5000 # how many to sample to load into memory at a time
# for fine-tuning
patience = 10000
patienceinc = 2
improvethresh = 1.005
# note, checks CV set every blocksize

if debug:
    epochs=5
    hlayers=[100]
    splitsize=0.2
    blocksize=1000

# load labels
labels = imt.readlabels(trfile)
m = len(labels)

# split to train and cv sets
trsplit = int(np.round(m*splitsize))
iterm=range(m)


alayers=[insz]
for q in hlayers:
    alayers.append(q)
alayers.append(outsz)
# train stacked autoencoder on unlabeled training data

blocksplits= range(0,trsplit,blocksize)
nblocks=len(blocksplits)-1
if trsplit % blocksize > 0:
    blocksplits.append(trsplit)

try:
    layermodels=pickle.load(open('data/pretune.p','r'))
except:
    layermodels=[]
    starttime=time.clock()
    for i in range(len(hlayers)):
        print('Training hidden layer %d'%i)
        # create denoising autoencoder, consider more options
        da=nnt.dAE(alayers[i],alayers[i+1],noise=noisel[i],errtype=etype)
        for j in range(epochs):
            print('Starting epoch %d...'%j)        
            for z in range(nblocks):
                # load up images
                if i==0:
                    x=imt.loadimgfromcsv(trfile,range(blocksplits[z],blocksplits[z+1]),colstart=1)/255.0
                else:
                    x=imt.loadimgfromcsv(acttmpfile,range(blocksplits[z],blocksplits[z+1]),colstart=0,headlines=0)
                nx=x.shape[0]
                for zz in range(nx):
                    da.GD(x[zz].reshape(x.shape[1],1),alpha=lr)
            epochend=time.clock()
            etime=(epochend-starttime)/60
            print('...finished in %2f minutes' %etime)
        if i<len(hlayers)-1:
            # save out activations
            print('Saving out activations...')
            writeto=acttmpfileroot+'_%d.csv'%i
            f=open(writeto,'wb')
            csvf=csv.writer(f)
            for c in range(nblocks):
                if i==0:
                    x=imt.loadimgfromcsv(trfile,range(blocksplits[c],blocksplits[c+1]),colstart=1)/255.0
                else:
                    x=imt.loadimgfromcsv(acttmpfile,range(blocksplits[c],blocksplits[c+1]),colstart=0,headlines=0)
                nx=x.shape[0]
                a=da.fprop(x.reshape(x.shape[1],nx))
                acts=a[1].reshape(nx,a[1].shape[0])
                csvf.writerows(acts)
            f.close()
            acttmpfile=writeto
            print('... done.')
        trtime=time.clock()
        elapsedtime = (trtime-starttime)/60
        print('Finished training layer %d, elapsed time: %2f min'%(i,elapsedtime))
        # save out weights
        layerw=da.getW(0,1) # 28*28 x nhiddenunits
        layermodels.append(da)
        if i==0:
            # visualize hidden layers by outputting image to file
            imt.squareImgPlot(layerw)
            imt.plt.savefig('data/W0.eps')

    pickle.dump(layermodels,open('data/pretune.p','wb')) # save out
    print('Finished Pre-tuning. Starting fine-tuning.')


# tune with softmax on labeled data, stopping gradient decent with best CV set prediction
sdAE=nnt.NN(alayers,actfun='S',errtype='SM')
# set weights to pretuned values
for l in range(len(layermodels)):
    sdAE.setW(layermodels[l].getW(0,1),l,1) # set W
    sdAE.setW(layermodels[l].getW(0,0),l,0) # set Bias

# keep the training set on disk and load cv set in memory
print('Loading CV set into memory...')
cvset = imt.loadimgfromcsv(trfile,iterm[trsplit:],colstart=1)/255.0
cvset=cvset.T
print('done')
cvlabels = labels[trsplit:]

bestscore = 0
loopflag = False
ep = 0
iter=0

while (ep<epochs) and not(loopflag):
    lcount=0
    print('Starting epoch %d'%ep)
    for z in range(nblocks):
        # load up images
        x=imt.loadimgfromcsv(trfile,range(blocksplits[z],blocksplits[z+1]),colstart=1)/255.0
        nx=x.shape[0]
        for zz in range(nx):
            sdAE.GD(x[zz].reshape(x.shape[1],1),labels[lcount],alpha=0.1)
            lcount+=1
            iter+=1
        # check model with CV set
        cvscore=sdAE.score(cvset,cvlabels)
        if cvscore >= bestscore:
            # best so far, but wait a bit
            if cvscore >= bestscore*improvethresh:
                # increase patience
                patience = np.maximum(patience,iter*patienceinc)
            bestscore = cvscore
        if iter >= patience:
            # done!
            loopflag=True
            print('Exiting optimization. CV score= %3f'%cvscore)
            break # out of block loop
    ep+=1   

# pickle the best model
pickle.dump(sdAE,open('data/finetuned.p','wb'))

# predict labels from test dataset
# process in all at once
testx=imt.loadimgfromcsv(testfile,colstart=0)/255.0
ntest=testx.shape[0]
preds = sdAE.predict(testx.T)
# output to .csv file labels
imt.writeoutpred(preds,range(ntest)+1,outfile,header=["ImageId","Label"])

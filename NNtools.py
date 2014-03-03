# Neural Network tools
import numpy as np

# a few activation functions:
def sigmoid(x):
    y=(1.0+np.exp(-x))**-1
    return y
    
def tanhfun(x):
    y=np.tanh(x)
    return y

class NN:
    def __init__(self,layers,W=None,L2reg=0.0,L1spar=0.0,errtype='SS',actfun='S',sigthr=0.5):
        layers=np.array(layers)
        self.layers=layers
        self.nlayers=len(layers)-1 # first layer is input layer
        self.L1spar=L1spar
        self.L2reg=L2reg
        self.errtype=errtype # SS: sum of squares error, CE= cross entropy, SM= softmax
        self.actfun=actfun # RL: rectified linear, S: sigmoid, T: tanh, L: linear
        self.sigthr=sigthr # for cross entropy: threshold for yes prediction
        inweights = layers[:-1]
        nunits = layers[1:]
        self.Wsize=inweights * nunits
        self.totalW = self.Wsize.sum()
        if not W:
            self.initW()
    
    def __repr__(self):
         return ('Neural Network Values: \nLayers: %s \nL2 Regularization: %g\n'
        'L1 Sparsity: %g\nCost Function: %s\nActivation Function: %s\n'
        'Sigmoid Threshold: %2f'%(self.layers,self.L2reg,self.L1spar,self.errtype,self.actfun,self.sigthr))
          
    def initW(self):
        # initialize weights to each layer of network between -r and r
        # self.layers is array with size of input layer, each hidden layer, and output layer
        # outputs initialized weights rolled into a single vector
        # rolled as W1,W2,...Wn,B1,B2,...Bn
        r  = np.sqrt(6) / np.sqrt(np.sum(self.layers[1:]))
        
        self.W=np.random.rand(self.totalW)*2*r-r
        self.W=np.append(self.W,np.zeros(sum(self.layers[1:]))) # set biases
    
    def getW(self,lnum,WorB):
        # retrieve weights from layer as ndarray input x nunits in size
        # input layer number (0 indexed)
        # and if it is a weight (1) or bias (0)
        
        if WorB: # getting weight
            Wstarts=np.append(0,self.Wsize.cumsum())
            Wout = self.W[Wstarts[lnum]:Wstarts[lnum+1]]
            Wout = Wout.reshape(self.layers[lnum],self.layers[lnum+1])
        else: # get bias
            Wstarts = np.append(self.totalW,self.totalW + self.layers[1:].cumsum())
            Wout = self.W[Wstarts[lnum]:Wstarts[lnum+1]]
        return Wout
        
    def setW(self,inW,lnum,WorB):
        # sets weights for layer from ndarray input x nunits in size
        # input layer number (0 indexed)
        # and if it is a weight (1) or bias (0)
        # note: to set weight from rolled W: use self.W=W
        # roll
        inW=inW.reshape(inW.size)
        
        if WorB: # setting weight
            Wstarts=np.append(0,self.Wsize.cumsum())
            self.W[Wstarts[lnum]:Wstarts[lnum+1]] = inW    
        else: # set bias
            Wstarts = np.append(self.totalW,self.totalW + self.layers[1:].cumsum())
            self.W[Wstarts[lnum]:Wstarts[lnum+1]] = inW

    def fprop(self,X):
        # perform forward propagation through NN
        # returns activations A for each layer as list of numpy ndarrays
        # note last layer is the linear activation
    
        # get number of examples
        if X.ndim>1:
            m=X.shape[1]
        else:
            m=1
        # perform forward pass through layers
        A=[X]
        for i in range(self.nlayers):
            # get the weights and multiply by activation
            wi=self.getW(i,1)
            a = np.dot(wi.T,A[i])
            # get bias and add to activation
            bi=self.getW(i,0)
            a+=bi.reshape(bi.size,1)
            # pass through activation function
            if i==(self.nlayers-1) or self.actfun=='L':
                A.append(a)
            else:
                if self.actfun=='S':
                    A.append(sigmoid(a))
                if self.actfun=='RL':
                    A.append(np.maximum(a,0))
                if self.actfun=='T':
                    A.append(tanhfun(a))
        return A    
            
    def bprop(self,A,grad):
        # go backward through hidden layers
        if A[0].ndim>1:
            m=A[0].shape[1]
        else:
            m=1
        layercount = range(len(A))
        revlayer = layercount[::-1][1:] #reversed count less last layer
        layererr = grad
        Wgrad = list(revlayer)
        Bgrad = list(revlayer) 
        for i in revlayer:
            # err in layer is:
            # (weights transpose * err in layer+1) element wise * 
            # deriv of layer activation wrt activation fxn (sigmoid)
        
            # get outgoing weights
            wi=self.getW(i,1)
            # err from layer n+1 
            # activation of layer i
            ai=A[i]
            # get derivative of activation function
            if self.actfun=='S':
                derivi=ai*(1-ai)
            if self.actfun=='T':
                derivi=1-ai**2
            if self.actfun=='L':
                derivi=1.0
            if self.actfun=='RL':
                derivi=(ai>0)*1.0
            Wgrad[i] = np.dot(ai,layererr.T)/m + self.L2reg * wi
            Bgrad[i] = (layererr.sum(axis=1))/m
            # if second layer then add sparsity err
            if i==1:
                sparerr = self.L1spar * np.sign(ai)
                layererr = (np.dot(wi,layererr)+sparerr) * derivi
            elif i>1:
                layererr = np.dot(wi,layererr) * derivi
            
        # string together gradients
        thetagrad=self.W*1.0
        wcount=0
        bcount=0
        for i in range(len(Wgrad)):
            nw=Wgrad[i].size
            thetagrad[wcount:nw+wcount]=Wgrad[i].reshape(nw)
            wcount+=nw
            nb=Bgrad[i].size
            thetagrad[self.totalW+bcount:self.totalW+bcount+nb]=Bgrad[i].reshape(nb)
            bcount+=nb
        return thetagrad
        
    def costFxn(self,X,Y):
        # determines the cost given input X and target Y
        # first to forward prop to get final activation
        # return cost, activations and gradient of last layer (to feed to back prop)
        
        A = self.fprop(X)
        # get number of examples
        if X.ndim>1:
            m=X.shape[1]
        else:
            m=1
        if self.errtype=='SS': # sum of squares err
            errcost = ((A[-1] - Y)**2).sum()/(2.0*m)
            grad = -(Y-A[-1])
            # TO DO: SS err with a rectified last layer
        elif self.errtype == 'CE': # cross entropy
            # pass through sigmoid
            Ahat = sigmoid(A[-1])
            errcost = (-Y*np.log(Ahat) - (1-Y)*np.log(1.0-Ahat)).sum()/m
            grad = -(Y-Ahat)
        elif self.errtype == 'SM': # softmax
            # Y is vector of m labels on {0,K-1}, make this a matrix of labels k x m
            # K is size of last layer
            Yhat = np.zeros((A[-1].shape[0],m))
            Yhat[Y,range(m)]=1.0
            # to prevent overflow subtract max
            maxA=A[-1].max(axis=0)
            Ahat = A[-1]-maxA.reshape(1,maxA.size)
            Ahat = np.exp(Ahat)
            Ahat = Ahat/Ahat.sum(axis=0)
            errcost = -(Yhat * np.log(Ahat)).sum()/m
            grad = -(Yhat-Ahat)
        else:
            print('Error type not recognized. Using sum of squares error\n')
            errcost = ((A - Y)**2).sum()/(2.0*m)
            grad = -(Y-A[-1])
            self.errtype='SS'
    
        # compute regularization cost
        regcost = 0.5 * self.L2reg * (self.W[:self.totalW]**2).sum()
    
        # L1 sparsity cost on first layer only 
        sparcost = self.L1spar*(1.0/m)*np.abs(A[1]).sum()
    
        # add up costs
        cost = errcost + regcost + sparcost
        return (cost,A,grad)
    
    def GD(self,X,Y,alpha=0.0001):
        # performs one step of gradient decent
        # X is n input features x m samples
        # for stochastic GD, m=1
        c,a,g=self.costFxn(X,Y)
        deltaw=self.bprop(a,g)
        self.W-=deltaw*alpha
        
    def predict(self,X):
        # predict Y given features X
        a=self.fprop(X)
        if self.errtype=='SS':
            yhat=a[-1]
        if self.errtype=='CE':
            yhat=sigmoid(a[-1])>=self.sigthr
        if self.errtype=='SM':
            yhat=np.argmax(a[-1],axis=0)
        
        return yhat
        
    def score(self,X,Y):
        # get accuracy of predictions
        preds=self.predict(X)
        correct=np.equal(preds,Y).mean()
        return correct

class dAE(NN):
    # denoising autoencoder layer
    def __init__(self,nfeat,hunits,noise=0.2,W=None,L2reg=0,L1spar=0,errtype='SS',actfun='S'):
        layers=np.array([nfeat,hunits,nfeat])
        self.layers=layers
        self.nlayers=len(layers)-1 # first layer is input layer, this is 2
        self.L1spar=L1spar
        self.L2reg=L2reg
        self.errtype=errtype # SS: sum of squares error, CE= cross entropy, SM= softmax
        self.actfun=actfun # RL: rectified linear, S: sigmoid, T: tanh, L: linear
        self.sigthr=0.5 # for cross entropy: threshold for yes prediction
        inweights = layers[:-1]
        nunits = layers[1:]
        self.noise=noise
        self.Wsize=inweights * nunits
        self.totalW = self.Wsize.sum()
        if not W: # can take weight vector as input, or initialize using tied weights
            self.initW()
            W0=self.getW(0,1)
            self.setW(W0.T.reshape(hunits,nfeat),1,1)
    
    def addNoise(self,X):
        # add corruption noise by setting a fraction of pixels to 0
        noisemask = 1.0*(np.random.rand(*X.shape)>=self.noise)
        Xhat = noisemask * X
        return Xhat
        
    def GD(self,X,alpha=0.01):
        # perform one step of gradient decent
        Xhat=self.addNoise(X)
        c,a,g=self.costFxn(Xhat,X)
        deltaw=self.bprop(a,g)
        # since weights are tied, need to add gradient of W0 and W1'
        wsize=self.layers[0] * self.layers[1]
        deltaW0=deltaw[:wsize].reshape(self.layers[0],self.layers[1])
        deltaW1=(deltaw[wsize:self.totalW].reshape(self.layers[1],self.layers[0])).T
        dw0and1=deltaW0+deltaW1
        deltaw[:wsize]=dw0and1.reshape(deltaW0.size)
        deltaw[wsize:self.totalW]=dw0and1.T.reshape(deltaW1.size)
        self.W-=deltaw*alpha


def numericalGradient(X,Y,J,e=1e-4):
    # compute numerical gradient as slope of J at theta values
    # J is a NN object with weights corresponding to theta
    theta=J.W.copy()
    perturb = np.zeros(np.size(theta))
    numgrad = np.zeros(np.size(theta))
    for p in range(np.size(theta)):
        perturb[p] = e
        J.W=theta-perturb
        loss1,A,g = J.costFxn(X,Y)
        J.W=theta+perturb
        loss2,A,g = J.costFxn(X,Y)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
        J.W=theta
    return numgrad

if __name__=='__main__':
    # run some debugging
    # defaults:
    l=[8,3,4,8]
    testnn = NN(l)
    x=np.random.rand(8,10)
    ng=numericalGradient(x,x,testnn)
    c,a,g=testnn.costFxn(x,x)
    testg=testnn.bprop(a,g)
    print('Cost Diff S SS: %g'%np.mean(abs(testg-ng)))
    testnn = NN(l,actfun='T')
    ng=numericalGradient(x,x,testnn)
    c,a,g=testnn.costFxn(x,x)
    testg=testnn.bprop(a,g)
    print('Cost Diff T SS: %g'%np.mean(abs(testg-ng)))
    testnn = NN(l,actfun='L')
    ng=numericalGradient(x,x,testnn)
    c,a,g=testnn.costFxn(x,x)
    testg=testnn.bprop(a,g)
    print('Cost Diff L SS: %g'%np.mean(abs(testg-ng)))
    
    l=[8,3,3]
    testsm= NN(l,errtype='SM')
    y=np.random.randint(3,size=10)
    ng=numericalGradient(x,y,testsm)
    c,a,g=testsm.costFxn(x,y)
    testg=testsm.bprop(a,g)
    print('Cost Diff S Softmax: %g'%np.mean(abs(testg-ng)))
    
    testsm= NN(l,actfun='T',errtype='SM')
    ng=numericalGradient(x,y,testsm)
    c,a,g=testsm.costFxn(x,y)
    testg=testsm.bprop(a,g)
    print('Cost Diff T Softmax: %g'%np.mean(abs(testg-ng)))
    
    
    l=[8,3,1]
    testce= NN(l,errtype='CE')
    y=np.random.randint(1,size=10)
    ng=numericalGradient(x,y,testce)
    c,a,g=testce.costFxn(x,y)
    testg=testce.bprop(a,g)
    print('Cost Diff S CE: %g'%np.mean(abs(testg-ng)))
    
    testce= NN(l,actfun='T',errtype='CE')
    ng=numericalGradient(x,y,testce)
    c,a,g=testce.costFxn(x,y)
    testg=testce.bprop(a,g)
    print('Cost Diff T CE: %g'%np.mean(abs(testg-ng)))
    
    s=dAE(8,3)
    print(s.W)
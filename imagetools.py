# Image processing tools for machine learning
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import csv


def loadImagePatches(imfile, imvar='IMAGES',patchsize=8,npatches=10000,edgebuff=5,scale0to1=True):
    # open .mat file containing images in a r x c x num images array
    # load patches that are patchsize x patchsize
    # normalize scale to 0 to 1 values
    imgdict = scipy.io.loadmat(imfile)
    imgarray = imgdict[imvar]
    # get dimentions
    r = imgarray.shape[0] - 2*edgebuff - patchsize
    c = imgarray.shape[1] - 2*edgebuff - patchsize
    nimg = imgarray.shape[2]
    
    # allocate random numbers and patches arrays
    patches = np.zeros([patchsize**2,npatches])
    randrow = np.random.randint(r,size=npatches) + edgebuff
    randcol = np.random.randint(c,size=npatches) + edgebuff
    randimg = np.random.randint(nimg,size=npatches)
    
    for i in range(npatches):
        r1 = randrow[i]
        r2 = r1+patchsize
        c1 = randcol[i]
        c2 = c1 + patchsize
        imi = randimg[i]
        patchi = imgarray[r1:r2,c1:c2,imi]
        patches[:,i] = patchi.reshape(1,patchsize**2)
    
    # normalize
    # subtract mean and scale by 3 stdev's
    patches -= patches.mean(0)
    pstd = patches.std() * 3
    patches = np.maximum(np.minimum(patches, pstd),-pstd) / pstd
    
    if scale0to1:
        # Rescale from [-1,1] to [0.1,0.9]
        patches = (patches+1) *  0.4 + 0.1
    
    return patches


def squareImgPlot(I):
    '''
    show n square images in a L x M array as single large panel image
    where each image is L**0.5 x L**0.5 pixels
    plotted image is M**0.5
    '''
    I = I - np.mean(I)
    (L, M)=I.shape
    sz=int(np.sqrt(L))
    buf=1
    if np.floor(np.sqrt(M))**2 != M :
        n=int(np.ceil(np.sqrt(M)))
        while M % n !=0 and n<1.2*np.sqrt(M): n+=1
        m=int(np.ceil(M/n))
    else:
        n=int(np.sqrt(M))
        m=n
    a=-np.ones([buf+m*(sz+buf)-1,buf+n*(sz+buf)-1])
    k=0
    for i in range(m):
        for j in range(n):
            if k>M: 
                continue
            clim=np.max(np.abs(I[:,k]))
            r1=buf+i*(sz+buf)
            r2=r1+sz
            c1=buf+j*(sz+buf)
            c2=c1+sz
            a[r1:r2,c1:c2]=I[:,k].reshape(sz,sz)/clim
            k+=1       
    h = plt.imshow(a,cmap='gray',interpolation='none',vmin=-1,vmax=1)

def loadimgfromcsv(filen,imgnums=None,headlines=1,colstart=0):
    '''
    Load images from a csv file organized with pixels in columns and each image a row
    returns numpy matrix with each row as an image vector
    options:
    imgnums = []: array of image indices to return. Empty returns all
    headlines = 1: number of header rows to skip
    colstart = 0: in which column (0 indexed) does pixel data start
    '''
    f = open(filen,'r')
    # skip headers
    for i in range(headlines):
        h=f.readline()
    # if imgnums is empty then get all columns
    if not(imgnums):
        imgs=f.readlines()
    # otherwise go through and find rows
    else:
        if not(np.iterable(imgnums)):
            imgnums=[imgnums]
        imgs=list()
        lastind = max(imgnums)
        count=0
        for i in enumerate(f):
            if count in imgnums:
                imgs.append(i[1]) # i is tuple with iterator value and values from f
            elif count>lastind:
                break
            count+=1
    # make into numpy array
    nimgs = len(imgs)
    # get first image
    im1=np.array(map(float,imgs[0].strip().split(',')[colstart:]))
    if nimgs>1:
        npix = len(im1)
        imgarr = np.zeros((nimgs,npix))
        count=0
        for s in imgs:
            imgarr[count][:]=np.array(map(float,s.strip().split(',')[colstart:]))
            count+=1
        if imgnums:
            imgarr=imgarr[np.argsort(np.argsort(imgnums))][:] # return in the order asked
    else:
        imgarr=im1
    f.close()
    return imgarr
    
def readlabels(filen,headlines=1,labelcol=0):
    f = open(filen,'r')
    # skip headers
    t=0
    for i in range(headlines):
        h=f.readline()
        t=f.tell()
    nlines=0
    for line in enumerate(f):
        nlines+=1
    labels=np.zeros(nlines)
    n=0
    f.seek(t)
    for line in enumerate(f):
        labels[n]=line[1].strip().split(',')[labelcol]
        n+=1
    f.close()
    return labels 
    
    

def writeoutpred(pred,testid,filename,header=["Id","Pred"]):
    # save out to .csv
    f = open(filename,'wb')
    csvf=csv.writer(f)
    csvf.writerow(header)
    csvf.writerows(zip(testid,pred))
    f.close()
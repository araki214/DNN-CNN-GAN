import numpy as np
import matplotlib.pyplot as plt
import chainer.optimizers as Opt
import chainer.functions as F
import chainer.links as L
import chainer.datasets as ds
import chainer.dataset.convert as con
from chainer import Variable,Chain,config,cuda

import princess as ohm
from chainer.iterators import SerialIterator as siter

train,test=ds.get_cifar10(ndim=3)
xtrain,ttrain=con.concat_examples(train)
xtest,ttest = con.concat_examples(test)

Dtrain,ch,Ny,Nx=xtrain.shape
print(Dtrain,ch,Ny,Nx)

C=ttrain.max()+1
H1=10
layers={}
layers["conv1"]=L.Convolution2D(ch,H1,ksize=3,stride=1,pad=1)
layers["bnorm1"]=L.BatchRenormalization(H1)
layers["l1"]=L.Linear(None,C)
NN=Chain(**layers)

def model(x):
    h=NN.conv1(x)
    h=F.relu(h)
    h=NN.bnorm1(h)
    h=F.max_pooling_2d(h,ksize=3,stride=2,pad=1)
    h=NN.l1(h)
    return h

gpu_device=0
cuda.get_device(gpu_device).use()
NN.to_gpu(gpu_device)
optNN=Opt.MomentumSGD()
optNN.setup(NN)

train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]
result=[train_loss,test_loss,train_acc,test_acc]

batch_size=5000
train_iter=siter(train,batch_size)

nepoch = 20
while train_iter.epoch < nepoch:
    batch = train_iter.next()
    xtrain,ttrain = con.concat_examples(batch)
    data=cuda.to_gpu([xtrain,xtest,ttrain,ttest])
    ohm.learning_classification(model,optNN,data,result,1)

ohm.plot_result2(result[0],result[1],"loss function","step","loss function",0.0,4.0)
ohm.plot_result2(result[2],result[3],"accuracy","step","accuracy",0.0,4.0)

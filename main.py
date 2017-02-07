import numpy as np
import gzip
import cPickle as pkl
from mlp import MLP
# import dill

f = gzip.open('mnist.pkl.gz', 'rb')
trainset, valset, testset = pkl.load(f)
f.close()

#------------------------------------------------------------------------------#

nhidden_layers = 2
nnodes = [200, 50]
activation = 'tanh'
classifier = MLP(nhidden_layers, nnodes, activation)

config = {
    'filename'    : 'layers_2_tanh_adam.log',
    'batchSize'   : 20,
    'max_epochs'  : 10,
    'optimizer'   : 'adam',
    'hyperParams' : [0.9, 0.9, 0.001]
}


classifier.train(trainset[0], trainset[1], valset[0], valset[1], config)
print '[' + config['filename']+'] Accuracy on test set is ' + str( classifier.test(testset[0], testset[1])) + '%'

#------------------------------------------------------------------------------#

nhidden_layers = 2
nnodes = [ 200, 50]
activation = 'relu'
classifier = MLP(nhidden_layers, nnodes, activation)

config = {
    'filename'    : 'layers_2_relu_adam.log',
    'batchSize'   : 20,
    'max_epochs'  : 10,
    'optimizer'   : 'adam',
    'hyperParams' : [0.9, 0.9, 0.001]
}


classifier.train(trainset[0], trainset[1], valset[0], valset[1], config)
print '[' + config['filename']+'] Accuracy on test set is ' + str( classifier.test(testset[0], testset[1])) + '%'

#------------------------------------------------------------------------------#

nhidden_layers = 2
nnodes = [ 200, 50]
activation = 'tanh'
classifier = MLP(nhidden_layers, nnodes, activation)

config = {
    'filename'    : 'layers_2_tanh_momn.log',
    'batchSize'   : 20,
    'max_epochs'  : 20,
    'optimizer'   : 'momentum',
    'hyperParams' : [0.9, 0.9, 0.001]
}


classifier.train(trainset[0], trainset[1], valset[0], valset[1], config)
print '[' + config['filename']+'] Accuracy on test set is ' + str( classifier.test(testset[0], testset[1])) + '%'
#------------------------------------------------------------------------------#

nhidden_layers = 2
nnodes = [ 200, 50]
activation = 'relu'
classifier = MLP(nhidden_layers, nnodes, activation)

config = {
    'filename'    : 'layers_2_relu_momn.log',
    'batchSize'   : 20,
    'max_epochs'  : 20,
    'optimizer'   : 'momentum',
    'hyperParams' : [0.9, 0.9, 0.001]
}


classifier.train(trainset[0], trainset[1], valset[0], valset[1], config)
print '[' + config['filename']+'] Accuracy on test set is ' + str( classifier.test(testset[0], testset[1])) + '%'


#------------------------------------------------------------------------------#

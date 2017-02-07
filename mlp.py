import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

class MLP(object):
    def __init__(self, nhidden_layers, nnodes, activation_function):
        '''
        nhidden_layers      : (int) number of layers
        nnodes              : (list) number of nodes per layer
        activation_function : (str) non-linear function applied after affine transformation
        '''
        assert nhidden_layers == len(nnodes)
        self.accumulated_momentum = None
        self.momentum1 = None
        self.momentum2 = None
        self.update_idx = 0
        self.rng = np.random.RandomState(1729)
        self.nhidden_layers = nhidden_layers
        self.nnodes = nnodes
        self.input_dim = 28*28
        self.no_of_classes = 10
        if activation_function == 'relu':
            self.activation_function = lambda x: x * ( x > 0)
            self.activation_derivative = lambda x: 1. * (x > 0)
        elif activation_function == 'tanh':
            self.activation_function = np.tanh
            self.activation_derivative = lambda x: 1.0 - np.tanh(x)**2
        else:
            raise NotImplementedError()
        self.build_model()






    def build_model(self):
        def get_params(self, input_dim, output_dim):
            W = np.asarray(self.rng.normal(loc=0.0,
                                           scale=0.001,
                                           size=(input_dim, output_dim)))
            b = np.asarray(self.rng.normal(loc=0.0,
                                           scale=0.001,
                                           size=(output_dim, )))
            return [W, b]
        self.params = []
        input_dim = self.input_dim
        output_dim = self.nnodes[0]
        allParams = get_params(self, input_dim, output_dim)
        self.params.append(allParams)
        for i in range(0, self.nhidden_layers-1):
            input_dim = output_dim
            output_dim = self.nnodes[i+1]
            self.params.append(get_params(self, input_dim, output_dim))
        input_dim = output_dim
        output_dim = self.no_of_classes
        self.params.append(get_params(self, input_dim, output_dim))





    def reduceSpace(self):
        self.accumulated_momentum = None
        self.momentum1 = None
        self.momentum2 = None
        self.update_idx = None
        self.gradParams = None
        self.input_vector = None
        self.outputs = None
        self.activ_derivative = None
        self.target = None




    def forward(self, input_vector):
        self.input_vector = input_vector
        def softmax(x):
            max_val = np.max(x)
            x = x - max_val
            return np.exp(x) / np.sum( np.exp(x), axis=0 )
        def softmax_derivative(vector):
            return np.diag(vector) - np.dot(np.expand_dims(vector, 1), np.expand_dims(vector, 0))
        self.outputs = []
        self.activ_derivative = []
        current_out = input_vector
        for param in self.params[:len(self.params)-1]:
            prev_out = current_out
            W, b = param
            affine_transform = np.dot(prev_out, W) + b
            current_out = self.activation_function(affine_transform)
            deriv_current_out = self.activation_derivative(affine_transform)
            self.outputs.append(current_out)
            self.activ_derivative.append( np.diag(deriv_current_out) )
        prev_out = current_out
        param = self.params[-1]
        W = param[0]
        b = param[1]
        affine_transform = np.dot(prev_out, W) + b
        current_out = softmax( affine_transform )
        self.outputs.append(current_out)
        self.activ_derivative.append(softmax_derivative(current_out))
        return current_out





    def cross_entropy_loss(self, target):
        self.target = np.zeros(self.no_of_classes)
        self.target[target] = 1
        probs = self.outputs[-1]
        if np.argmax(probs) == target:
            self.hit = 1
        else:
            self.hit = 0
        self.loss = -1 * np.log(probs[target])
        return self.loss




    def backward(self, target=None):
        self.gradParams = []
        if not target == None:
            self.target = np.zeros(self.no_of_classes)
            self.target[target] = 1
        input_sequence = [self.input_vector] + self.outputs[0:len(self.outputs)-1]
        dJ_dX = self.outputs[-1] - self.target    # derivative of softmax layer w.r.t. cost
        for param, gdY, X in zip(self.params[::-1], self.activ_derivative[::-1],  input_sequence[::-1]):                            # going in reverse order
            dJ_dY = dJ_dX                         # previous dJ_dX is current dJ_dY
            W = param[0]
            dJ_dX = np.dot(np.dot(W, gdY), dJ_dY)
            dJ_dW = np.dot(np.expand_dims(X, axis=1), np.expand_dims(np.dot(gdY, dJ_dY), axis=0))
            dJ_db = np.dot(gdY, dJ_dY)
            self.gradParams = [[dJ_dW, dJ_db]] + self.gradParams
        return dJ_dX





    def updateParams(self, hyperParams, optimizer):
        '''
        This function is used for updating parameters.
        Expects the following input
        hyperParams : (list) a list of hyperParams for the type of optimizer
        optimizer   : (str) 'momentum' or 'adam'
            momentum:
             Takes the following params as a list:
             learningRate : hyperParams[0]
             gamma        : hyperParams[1]

            adam:
             Takes the following params as a list:
             beta1 : hyperParams[0]
             beta2 : hyperParams[1]
             alpha : hyperParams[2]
            '''
        def momentum(hyperParams):
            learningRate = hyperParams[0]
            gamma        = hyperParams[1]
            if self.accumulated_momentum == None:
                self.accumulated_momentum = []
                for gparam in self.gradParams:
                    self.accumulated_momentum.append([np.copy(gparam[0]),
                                                      np.copy(gparam[1])] )
            else:
                self.accumulated_momentum =  map(lambda x, y: [gamma * x[0] - (1-gamma) * y[0],
                                                               gamma * x[1] - (1-gamma) * y[1]],
                                                 self.accumulated_momentum, self.gradParams )

            self.params = map(lambda x, y: [x[0] - learningRate*y[0], x[1] - learningRate*y[1]],
                              self.params, self.accumulated_momentum)

        def adam(hyperParams):
            beta1 = hyperParams[0]
            beta2 = hyperParams[1]
            alpha = hyperParams[2]
            epsilon = 10e-8

            self.update_idx += 1
            i_t = self.update_idx

            if self.momentum1 == None:
                self.momentum1 = []
                self.momentum2 = []
                for g in self.gradParams:
                    self.momentum1.append([np.copy(g[0]),
                                           np.copy(g[1])])
                    self.momentum2.append([np.copy(np.square(g[0])),
                                           np.copy(np.square(g[1]))])
            else:
                self.momentum1 = map( lambda x, y: [beta1 * x[0] + (1 - beta1)*y[0],
                                                    beta1 * x[1] + (1 - beta1)*y[1]],
                                      self.momentum1, self.gradParams)
                self.momentum2 = map( lambda x, y: [beta2 * x[0] + (1 - beta2)*(np.square(y[0])),
                                                    beta2 * x[1] + (1 - beta2)*(np.square(y[1]))],
                                      self.momentum2, self.gradParams)

            m_t = map(lambda x: [x[0]/(1 - beta1**i_t), x[1]/(1 - beta1**i_t)],
                       self.momentum1)
            v_t = map(lambda x: [x[0]/(1 - beta2**i_t), x[1]/(1 - beta2**i_t)],
                       self.momentum2)
            self.params = map( lambda theta, m, v:
                               [theta[0] - alpha * np.divide(m[0], np.sqrt(v[0] + epsilon)),
                                theta[1] - alpha * np.divide(m[1], np.sqrt(v[1] + epsilon))],
                               self.params, m_t, v_t)

        if optimizer == 'momentum':
            momentum(hyperParams)
        else:
            adam(hyperParams)






    def get_numerical_gradParams(self, input_vector, target):
        '''
        To verify the calculated gradients are close to real gradients
        '''
        epsilon = 1e-10
        self.forward(input_vector)
        self.backward(target)  # We have gradParams here
        calculated_gradParams = []
        squared_error = []

        def calculate_W_grad(layer, epsilon, input_vector, target):
            n, m = self.params[layer][0].shape
            gradW = np.ndarray(self.params[layer][0].shape)
            squared = np.ndarray(self.params[layer][0].shape)
            for i in range(0, n):
                for j in range(0, m):
                    orig_value = self.params[layer][0][i][j]
                    self.params[layer][0][i][j] += epsilon
                    self.forward(input_vector)
                    J_plus = self.cross_entropy_loss(target)
                    self.params[layer][0][i][j] -= 2*epsilon
                    self.forward(input_vector)
                    J_minus = self.cross_entropy_loss(target)
                    grad_val = (J_plus - J_minus)/(epsilon*2)
                    gradW[i][j] = grad_val
                    squared[i][j] = (grad_val - self.gradParams[layer][0][i][j])**2
                    self.params[layer][0][i][j] = orig_value
            return gradW, squared

        def calculate_b_grad(layer, epsilon, input_vector, target):
            m = self.params[layer][1].shape[0]
            gradb = np.ndarray(self.params[layer][1].shape)
            squared = np.ndarray(self.params[layer][1].shape)
            for i in range(0, m):
                orig_value = self.params[layer][1][i]
                self.params[layer][1][i] += epsilon
                self.forward(input_vector)
                J_plus = self.cross_entropy_loss(target)
                self.params[layer][1][i] -= 2*epsilon
                self.forward(input_vector)
                J_minus = self.cross_entropy_loss(target)
                grad_val = (J_plus - J_minus)/(epsilon*2)
                gradb[i] = grad_val
                squared[i] = ( grad_val - self.gradParams[layer][1][i] )**2
                self.params[layer][1][i] = orig_value
            return gradb, squared

        for i in tqdm( xrange(0, len(self.params)) ):
            # calculate W grads
            gradW, squaredW = calculate_W_grad(i, epsilon, input_vector, target)
            # calculate b grads
            gradb, squaredb = calculate_b_grad(i, epsilon, input_vector, target)
            calculated_gradParams.append([gradW, gradb])
            squared_error.append([squaredW, squaredb])

        return calculated_gradParams, squared_error








    def plot_grads(self, fname, input_vector=None, target=None):
        '''
        To plot the calculated gradients and check if they are close to real gradients
        '''
        self.forward(input_vector)
        self.backward(target)  # We have gradParams here
        calculated_gradParams, squared_error =  self.get_numerical_gradParams(input_vector, target)
        X = []
        Y = []
        for squared in squared_error:
            squared_gradW, squared_gradb = squared
            n, m = squared_gradW.shape
            for j in range(0, m):
                for i in range(0, n):
                    X.append(len(X))
                    Y.append(squared_gradW[i][j])
                X.append(len(X))
                Y.append(squared_gradb[j])

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.hist(X, Y, label='Squared Difference')
        # ax.legend()
        # ax.set_ylabel('gradient values')
        # ax.set_xlabel('parameter indexes')
        # fig.savefig(fname)
        plt.figure()
        plt.hist(Y)
        plt.title('Squared Difference')
        plt.xlabel('squared difference')
        plt.ylabel('frequency')
        plt.savefig(fname)






    def test( self, test_data, test_labels):
        hits = 0
        testsize = test_labels.shape[0]
        for inp, target in zip(test_data, test_labels):
            self.forward(inp)
            self.cross_entropy_loss(target)
            hits += self.hit
        return (100.0*hits)/testsize





    def train(self, train_data, train_labels, val_data, val_labels, config):
        '''
        config : a dictionary with following entries
         filename   : name of file to store results
         batchSize  : batchSize for minibatch
         max_epochs : number of epochs to run
         optimizer  : 'momentum' or 'adam'
         hyperParams:
          momentum:
             learningRate : hyperParams[0]
             gamma        : hyperParams[1]
          adam:
             beta1 : hyperParams[0]
             beta2 : hyperParams[1]
             alpha : hyperParams[2]
        '''
        train_size = train_labels.shape[0]
        val_size = val_labels.shape[0]
        shuffle_train = np.random.permutation(np.arange(0, train_size))
        shuffle_val   = np.random.permutation(np.arange(0, val_size))

        filename       = config['filename']
        hyperParams    = config['hyperParams']
        optimizer      = config['optimizer']
        minibatch_size = config['batchSize']
        max_epochs     = config['max_epochs']
        f = open(filename, 'w')
        assert minibatch_size > 0

        def accumulate_grads( g1, g2 ):
            if g1 == []:
                return g2
            else:
                return map( lambda x, y: [x[0] + y[0],
                                        x[1] + y[1]],
                            g1, g2)
        loss_train = 0
        def TrainStep():
            loss_train = 0
            hits = 0
            for i in range(0, train_size, minibatch_size ):
                local_gradParams = []
                local_loss = 0
                for j in range(0, minibatch_size):
                    index = shuffle_train[i + j]
                    self.forward( train_data[index])
                    local_loss += self.cross_entropy_loss( train_labels[index] )
                    hits += self.hit
                    self.backward()
                    local_gradParams =  accumulate_grads(local_gradParams, self.gradParams)

                # averaging gradParams
                self.gradParams = map(lambda x:[x[0]/minibatch_size,
                                                 x[1]/minibatch_size],
                                      local_gradParams)
                self.updateParams(hyperParams, optimizer)
                loss_train += local_loss
            return [loss_train/train_size, (hits*100.0)/train_size]

        def ValidationStep():
            loss_val = 0
            hits = 0
            for i in range(0, val_size):
                self.forward( val_data[ shuffle_val[i] ] )
                loss_val += self.cross_entropy_loss(val_labels[ shuffle_val[i] ])
                hits += self.hit
            return [loss_val/val_size, (hits*100.0)/val_size]

        for epoch in tqdm( xrange(0,max_epochs) ):
            print 'going for a train step'
            loss_train, accuracy_train = TrainStep()
            print 'going for ValidationStep'
            loss_val, accuracy_val = ValidationStep()
            f.write( str(loss_train)+', '+str(accuracy_train)+','+str(loss_val)+','+str(accuracy_val)+'\n')

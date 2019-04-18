#!/usr/bin/env python3
# Numpy implementation of fully connected neural network model on CIFAR 10 dataset

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_dimensions):
        self.layer_num = layer_dimensions.shape[0]  # number of total layers
        self.dim = layer_dimensions  # all layers' dimensions
        # weight, bias, g(z), output z, dropout mask.
        self.W, self.b, self.A, self.Z, self.M = [], [], [], [], []
        
        # initialize the weight and bias
        for i in range(self.layer_num - 1):
            self.W.append(0.002*np.random.rand(self.dim[i], self.dim[i+1]))
            self.b.append(np.zeros(self.dim[i+1]))
        
        self.p = 0  # initial dropout                          
        self.batch_size = 0 # initial batch size
        self.model = 1 # train model
        
    def affineForward(self, Z, W, b):
        A = Z.dot(W) + b
        return A
    
    def activationForward(self, Z):
        # ReLU activation function
        y = np.maximum(Z, 0)
        return y
    
    def Softmax(self, Z):
        A = np.exp(Z)
        K = np.diag(1/np.sum(A, 1))
        y = K.dot(A)
        return y

    def stable_Softmax(self, X):
        Z = (X.transpose()-np.max(X,1)).transpose()
        A = np.exp(Z)
        K = np.diag(1/np.sum(A, 1))
        y = K.dot(A)
        return y
    
    def forwardPropagation(self, X):
        self.A, self.Z, self.M = [],[],[]  # firstly empty the lists
        self.A.append(X)  
        W, M = self.dropout(self.W[0], self.p) # dropout
        Z = self.affineForward(X, W, self.b[0])
        self.Z.append(Z)  # storage the A,Z to compute gradient
        self.M.append(M)  # store M to use in the backpropagation
        
        for i in range(1, self.layer_num - 1): # forward in loop
            A = self.activationForward(Z)
            self.A.append(A)
            W, M = self.dropout(self.W[i], self.p)
            Z = self.affineForward(A, W, self.b[i])
            self.Z.append(Z)
            self.M.append(M)
            
        return self.Softmax(Z)
        
        
    def costFunction(self, AL, Y): # Cross-Entropy loss
        TINY = 10e-15 # avoid log(0)
        M = Y.shape[0]
        N = Y.shape[1]  # AL: M*N, Y: M*N
        # Loss = -y*log(y_hat), y is ground truth
        Cross_Entropy = - np.sum(Y * np.log(AL + TINY)) / M
        return Cross_Entropy
    
    def affineBackward(self, dZ_pre, cache):
        return 0
    
    def activationBackward(self, dZ_cur, W_cur, Z_pre):
        gZ_pre = 1*(Z_pre>0)  # g'(Z_(n-1))
        # dZ_(n-1) = dZ_n * W_n * g'(Z_(n-1))
        dZ_pre = dZ_cur.dot(np.transpose(W_cur))*gZ_pre
        return dZ_pre
    
    def backPropagation(self, AL, Y):
        M = Y.shape[0]        # number of batch
        Mask = self.M
        H = self.layer_num-1  # number of layers
        dW, db = [], []
        
        # dL/dZ_last = softmax(Z_last)-Y
        # AL = A_last = softmax(Z_last), from forwardProp
        dZ_last = AL - Y
        # note that the layer of AL is H
        A_pre = self.A[H-1]
        
        # dW_n = dZ_n * A_(n-1), M is batch size
        dW_last = dZ_last.transpose().dot(A_pre) / M
        db_last = np.sum(dZ_last,0) / M
        dW.append(dW_last)
        db.append(db_last)
        dZ_cur = dZ_last
        
        for i in range(1, H):
            W_cur = self.W[H-i] * Mask[H-i]
            Z_pre = self.Z[H-i-1]
            dZ_pre = self.activationBackward(dZ_cur, W_cur, Z_pre) # dZ_n
            
            dZ_cur = dZ_pre # dZ_(n-1)
            A_pre = self.A[H-i-1]
            # a trick to figure out gradient vanish
            # update batch_size times for previous weight to avoid vanishing
            dW_cur = dZ_cur.transpose().dot(A_pre) / M * M
            db_cur = np.sum(dZ_cur,0)/ M * M
            dW.append(dW_cur)
            db.append(db_cur)
            
        return dW, db
    
    def updateParameters(self, dW, db, alpha):
        Mask = self.M    # dropout mask
        layer_num = self.layer_num
        for i in range(layer_num-1):
            self.W[i] = self.W[i] - alpha*dW[layer_num-i-2].transpose()*Mask[i]
            self.b[i] = self.b[i] - alpha*db[layer_num-i-2].transpose()
    
    def dropout(self, A, prob):
        self.p = 0.4
        M = np.random.rand(A.shape[0], A.shape[1])
        if self.model == 1:  # train model
            M = (M > prob) * 1.0
            M /= (1 - prob)
            W = A * M   # use a new variable
            return W, M
        elif self.model == 0: # test model
            return A, M
    
    def predict(self, X_new):
        X = np.reshape(X_new.numpy(), (1, 3*32*32))
        output = self.forwardPropagation(X)
        pred = np.argmax(output, 1)
        return pred
    
    def train(self, X_train, X_val, y_train, y_val, 
              iters, alpha, batch_size, dropout_prob):
        
        self.batch_size = batch_size    # input batch size
        self.p = dropout_prob           # input dropout probability
        Batch = int(train_size/batch_size) + 1 # number of batch
        
        for epoch in range(iters):
            self.model = 1      # begin to train
            loss_total = 0      # total loss
            correct = 0         # total correct number
            if epoch >= 3:      # adaptive changing learning rate
                alpha = 0.001
            
            for i in range(Batch):
                X = X_train[batch_size*i:batch_size*(i+1)]
                Y = y_train[batch_size*i:batch_size*(i+1)]
                M = Y.shape[0]  # current batch size
                output = self.forwardPropagation(X)
                loss = self.costFunction(output, Y)

                loss_total += loss * M
                dW, db = nn.backPropagation(output, Y)
                  
                label = np.argmax(Y,1)
                pred = np.argmax(output,1)
                accuracy = 100.0 * np.sum(pred == label) / M
                nn.updateParameters(dW, db, alpha)
                if i % 100 == 0:
                    print('Epoch:{:d}\t batch:{:d}\t Loss:{:0.6f}\t Accuracy:{:0.2f}%'.format(
                        epoch, i, loss, accuracy))
                    
            # Average Training Loss
            print('Epoch:{:d}\t Average Training Loss:{:0.4f}'.format(
                epoch, loss_total / train_size))
            
            # Testing
            self.model = 0
            correct = 0
            # Train Accuracy
            for i in range(Batch):
                X = X_train[batch_size*i:batch_size*(i+1)]
                Y = y_train[batch_size*i:batch_size*(i+1)]
                output = self.forwardPropagation(X)
                label = np.argmax(Y,1)
                pred = np.argmax(output,1)
                correct += 1.0 * np.sum(pred == label)
            
            accuracy = 100* correct / y_train.shape[0]
            print('Epoch:{:d}\t Train Accuracy: {:0.2f}%'.format(epoch, accuracy))
        
            # Validation Accuracy
            correct = 0
            val_Batch = int(val_size/batch_size) + 1
            for i in range(Batch):
                X = X_val[batch_size*i:batch_size*(i+1)]
                Y = y_val[batch_size*i:batch_size*(i+1)]
                output = self.forwardPropagation(X)
                pred = np.argmax(output,1)
                correct += 1.0 * np.sum(pred == Y)
            
            accuracy = 100* correct / y_val.shape[0]
            print('Epoch:{:d}\t Validation Accuracy: {:0.2f}%'.format(epoch, accuracy))
            
# Load Data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_val_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

train_set, val_set = torch.utils.data.random_split(train_val_set, [45000, 5000])

# Check shape
train_size = len(train_set)
val_size = len(val_set)
test_size = len(test_set)

# Create numpy data
train_image = np.zeros([train_size, 3, 32, 32])
train_label = np.zeros(train_size)
val_image = np.zeros([val_size, 3, 32, 32])
val_label = np.zeros(val_size)
test_image = np.zeros([test_size, 3, 32, 32])
test_label = np.zeros(test_size)

# Transform Tensor to Numpy
for i in range(len(train_set)):
    train_image[i] = train_set[i][0].numpy()
    train_label[i] = train_set[i][1]
for i in range(len(val_set)):
    val_image[i] = val_set[i][0].numpy()
    val_label[i] = val_set[i][1]
for i in range(len(test_set)):
    test_image[i] = test_set[i][0].numpy()
    test_label[i] = test_set[i][1]

# Check shape
print(train_image.shape)
print(train_label.shape)
print(val_image.shape)
print(val_label.shape)
print(test_image.shape)
print(test_label.shape)

# Convert to flattened numpy arrays
M = train_label.shape[0]
N = 10

X_train = np.reshape(train_image, (45000, 3*32*32))
Y_train = np.zeros([M,N])
for i in range(M):
    Y_train[i, int(train_label[i])] = 1

X_val = np.reshape(val_image, (5000, 3*32*32))
Y_val = val_label

X_test = np.reshape(test_image, (10000, 3*32*32))
Y_test = test_label

# Check shape
print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)

# Training
# NN architecture: 3072 -> 2048 -> 2048 -> 10
nn = NeuralNetwork(np.array([3072,2048,2048,10]))
nn.train(X_train=X_train, X_val=X_val, y_train=Y_train, 
         y_val=Y_val, iters=5, alpha=0.01, batch_size=64, dropout_prob=0.4)

# Visualize and Prediction
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy() 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
# test image 1
X_test = test_set[3][0]
y_test = test_set[3][1]
imshow(X_test)

Classes = np.array(['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck'])
pred = nn.predict(X_test)
print('The image is', Classes[y_test])
print('The prediction is', Classes[pred[0]])

# test image 2
X_test = test_set[15][0]
y_test = test_set[15][1]
imshow(X_test)

Classes = np.array(['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck'])
pred = nn.predict(X_test)
print('The image is', Classes[y_test])
print('The prediction is', Classes[pred[0]])

# test image 3
X_test = test_set[23][0]
y_test = test_set[23][1]
imshow(X_test)

Classes = np.array(['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck'])
pred = nn.predict(X_test)
print('The image is', Classes[y_test])
print('The prediction is', Classes[pred[0]])



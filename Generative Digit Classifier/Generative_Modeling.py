import torch
import numpy as np
from torch.distributions import normal
from torch.distributions import multivariate_normal


''' Read data from txt as tensors'''
def readData(trainFile, testFile, valFile):
    
    # read training data
    train_data = np.loadtxt(trainFile)
    train_vec = train_data[:,:-1]
    train_lab = train_data[:,-1]
    
    # read testing data
    test_data = np.loadtxt(testFile)
    test_vec = test_data[:,:-1]
    test_lab = test_data[:,-1]
    
    # read validation data
    val_data = np.loadtxt(valFile)
    val_vec = val_data[:,:-1]
    val_lab = val_data[:,-1]
    
    return torch.tensor(train_vec), torch.tensor(train_lab), torch.tensor(test_vec), torch.tensor(test_lab), \
        torch.tensor(val_vec), torch.tensor(val_lab)


# feature extractors
'''
    Feature extractor 0 --- only for univariate gaussian
    input: vectors of extended image (None, 784)
    output: sum of all pixel values for each image (None, 1)
    '''
def fe0(X):
    return torch.sum(X,dim=1)

'''
    Feature extractor 1 --- only for multivariate gaussian
    input: vectors of extended image (None, 784)
    output: number of nonzero pixels on each row & each column.  (None, 56)
    Remember that the input image is of size 28*28
'''
def fe1(X, threshold=0):
    X_row = (X>threshold).reshape(-1,28,28).sum(dim=1)
    X_col = (X>threshold).reshape(-1,28,28).sum(dim=2)
    return torch.cat([X_row,X_col],dim=1).double()

'''
    Feature extractor 2 --- only for multivariate gaussian
    input: vectors of extended image (None, 784)
    output: input                    (None, 784)
'''
def fe2(X):
    return X    

fe = [fe0,fe1,fe2]

class GNB:
    '''
    Constructor
    input: uni - set as True for univariate model
           ID_FE - set as 2 for fe2, otherwise don't change
    '''
    def __init__ (self, uni=True, ID_FE=0):
        self.uni = uni
        # do not change these
        self.prior = None       # list of the priors. Made from empirical counts
        self.dists = None       # list of Gaussian distributions likelihood. size = number of classes
        self.num_c = 0          # number of classes
        self.ID_FE = max(0 if self.uni else 1, ID_FE) # select feature extractor
    
    '''
    Train your model
    input: X - training data
           y - training labels
           uni - set as True for univariate model
    '''
    def train(self, X, y):
        self.dists = list()
        uniq_y = torch.unique(y)                      # get all labels
        self.prior = torch.zeros(uniq_y.shape[0])     # initialize prior P(Y) to zeros
        self.num_c = uniq_y.shape[0]                  # number of classes   

        for idx in range(self.num_c):           # for each label
            mask = torch.eq(y,idx)
            x_by_label = X[mask]           # select all observations with label

            self.prior[idx] = float( len(x_by_label) )/ len(y)              ##### Estimate model's prior P(Y) 

            x = fe[self.ID_FE](x_by_label)      # extract the feature vector

            if self.uni: ##### Univariate gaussian 

                #####       Estimate the univariate Gaussian conditional distributions.          
                #####       Compute the empirical mean and variance. Then call PyTorch's "Normal" 
                #####       function with the correct arguments to build the distribution. 
                
                x_mean= torch.mean(x , dim=0)                  #####  calculate the mean value 
                x_var = torch.std( x , dim=0 )                 ##### calculate the variance
                gaussian_dist = normal.Normal( x_mean , x_var )   

            else: ##### Multivariate Gaussian.

                #####       Estimate the multivariate Gaussian conditional distributions.         
                #####       Call PyTorch's "MultivariateNormal" function to biuld    
                #####       a multivariate normal distribution.
                
                # transpose the input vector for ease of usage
                x = torch.transpose( x,0,1  )   
                x_mean= torch.mean( x , dim=-1 )                  ##### estimate the mean vector

                # measure covarince
                N = x.shape[-1]
                x = x - x_mean.unsqueeze(-1) 
                x_cov = torch.eye(56 , dtype=torch.float64) + 1/(N-1) * x @ x.T                ##### estimate the covariance matrix
                gaussian_dist = multivariate_normal.MultivariateNormal( x_mean , x_cov  )         
            
            self.dists.append(gaussian_dist)
        return

    '''
        Use Bayes rule to predict on one sample
    '''
    def predict(self, x):
        x = fe[self.ID_FE](torch.reshape(x,(1,-1)))
        result = [self.dists[i].log_prob(x) + torch.log(self.prior[i]) for i in range(self.num_c)]
        return torch.argmax(torch.tensor(result))

    '''
        Evaluate classification accuracy
    '''
    def evaluate(self, X, y):
        correct = 0
        for i in range(X.shape[0]):
            if self.predict(X[i]) == y[i].long():
                correct = correct + 1
        print('The classification accuracy is {:.3f}'.format(correct/X.shape[0]))
        return


if __name__ == "__main__":

    train_vec, train_lab, test_vec, test_lab, val_vec, val_lab = readData('train.txt','test.txt','validate.txt')   

    model = GNB(uni=False)  #create model; uni - set as False for multivariate model

    model.train(train_vec,train_lab)       # Call univariate train function to estimate your model's parameters from the training data
    #    train(train_vec,train_lab,False)  # Call multivariate train function to estimate your model's parameters from the training data

    #    evaluate(train_vec, train_lab)    # Call evaluate function to compute classification error on the training data.
    model.evaluate(val_vec, val_lab)       # Call evaluate function to compute classification error on the validation data.
    #    evaluate(test_vec, test_lab)      # Call evaluate function to compute classification error on the test data.





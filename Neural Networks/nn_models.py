"""
baselines.py: contains all network structure definition
including layers definition and forward pass function definition
"""
# PyTorch and neural network imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import numpy as np

# set the randomness to keep reproducible results
torch.manual_seed(0)
np.random.seed(0)

# input size to your mlp network
mlp_input_size = 784 
# final output size of  mlp network (output layer)
mlp_output_size = 10   
# width of hidden layer
mlp_hidden_size = 12

class BaselineMLP(nn.Module):
    def __init__(self):
        """
        A multilayer perceptron model
        Consists of one hidden layer and 1 output layer (all fully connected)
        """
        super(BaselineMLP, self).__init__()
        # a fully connected layer from input layer to hidden layer
        # mlp_input_size denotes how many input neurons we have
        # mlp_hiddent_size denotes how many hidden neurons we have
        self.fc1 = nn.Linear(mlp_input_size, mlp_hidden_size)
        # a fully connected layer from hidden layer to output layer
        # mlp_output_size denotes how many output neurons you have
        self.fc2 = nn.Linear(mlp_hidden_size, mlp_output_size)
    
    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying 
        logistic activation function after hidden layer.
        """
        # pass X from input layer to hidden layer
        out = self.fc1(X)
        # apply an activation function to the output of hidden layer
        out = torch.sigmoid(out)
        # pass output from hidden layer to output layer
        out = self.fc2(out)
        # return the feed forward output
        return out


class BaselineCNN(nn.Module):
    def __init__(self):
        """
        A basic convolutional neural network model for baseline comparison.
        Consists of one Conv2d layer, followed by 1 fully-connected (FC) layer:
        conv1 -> fc1 (outputs)
        """
        super(BaselineCNN, self).__init__()
        # define different layers


    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying 
        non-linearities after each layer.
        
        Note that this function *needs* to be called "forward" for PyTorch to 
        automagically perform the forward pass.

        You may need the function "num_fc_features" below to help implement 
        this function
        
        Parameters: X --- an input batch of images
        Returns:    out --- the output of the network
        """
        # define the forward function
        out = X
        return out

    """
    Count the number of flattened features to be passed to fully connected layers
    Parameters: inputs --- 4-dimensional [batch x num_channels x conv width x conv height]
                            output from the last conv layer
    Return: num_features --- total number of flattened features for the last layer
    """
    def num_fc_features(self, inputs):
        
        # Get the dimensions of the layers excluding the batch number
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

"""
new neural network (customized)
"""
class my_network(nn.Module):

    def __init__(self):
        """
        A multilayer perceptron model
        Consists of one hidden layer and 1 output layer (all fully connected)
        """
        super( my_network , self).__init__()

        # input size to mlp network
        mlp_input_size = 784 # after image linearized 28*28 image become 784
        # final output size of your mlp network (output layer)
        mlp_output_size = 10   # since mnist having 10 classes output size should be 10
        # width of your hidden layer
        mlp_hidden_size = 256

        # linear layer --> 784 to 256
        self.fc1 = nn.Linear(mlp_input_size, mlp_hidden_size)
        # adding batchnormalization layer  which helps to normalize the activations
        self.fc1_bn = nn.BatchNorm1d(  mlp_hidden_size )
        self.fc1_drop = nn.Dropout( p = 0.3 )
        # add new linear layer with hidden size of 512
        self.fc2 = nn.Linear( mlp_hidden_size , mlp_hidden_size )
        # add another batchnormlaiztion layer
        self.fc2_bn = nn.BatchNorm1d(  mlp_hidden_size )
        # adding dropout layer to reduce overfitting
        self.fc2_drop = nn.Dropout( p = 0.4 )
        # define the final classification layer
        self.fc3 = nn.Linear( mlp_hidden_size  , mlp_output_size)
    
    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying 
        logistic activation function after hidden layer.
        """
        # pass X from input layer to hidden layer
        out = self.fc1(X)
        # apply an relu activation function to the output of hidden layer
        out = torch.relu(out)
        # pass the output into batchnorm layer
        out = self.fc1_bn( out )
        # pass the output from dropout layer to reduce the overfitting
        out = self.fc1_drop( out )
        # pass output from hidden layer to second new hidden layer
        out = self.fc2( out )
        # apply an relu activation function to the output of hidden layer
        out = torch.relu(out)
        # pass the output from batchnormalization layer
        out = self.fc2_bn( out )
        # pass the output from dropout layer
        out = self.fc2_drop( out )
        # pass throught the final classification layer
        out = self.fc3( out )

        # return the feed forward output
        
        return out

# torch imports
from torch.nn.functional import sigmoid
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # define any initial layers, here      
        self.input = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.relu_1 = nn.ReLU()
        self.lin_1 = nn.Linear(in_features=self.hidden_dim, out_features=120)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(in_features=120, out_features=self.output_dim)
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, input_):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        x = self.input(input_)
        x = self.relu_1(x)
        x = self.lin_1(x)
        x = self.prelu(x)
        x = self.out(x)
        return sigmoid(x)
    
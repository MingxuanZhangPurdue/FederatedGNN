import torch
import scipy
import copy
import scipy.linalg
import numpy as np
import torch.nn as nn



class MLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, bias=False):
        
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        
    def forward(self, X):
        
        """
        X: [batch_size, input_dim], float tensor
        """
        
        if (X.type() != 'torch.FloatTensor'):
            X = X.type(torch.FloatTensor)
        
        X = F.relu(self.linear1(X))
        H = self.linear2(X)
        
        # H: [batch_size, output_dim], the feature representation
        return H
    
    
class LR(nn.Module):
    
    def __init__(self, input_dim, output_dim, bias=False):
        
        super(LR, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
    def forward(self, X):
        
        """
        X: [batch_size, input_dim], float tensor
        """
        
        if (X.type() != 'torch.FloatTensor'):
            X = X.type(torch.FloatTensor)
        
        H = self.linear(X)
        
        # H: [batch_size, output_dim], the feature representation
        return H

def calculate_Atilde(A, K, alpha):
    
    
    """
    A: adjacent matrix, numpy array, [N, N]
    K: number of power iterations, scalar
    alpha: jump probability, scalar
    """
    
    
    # Number of nodes in this graph
    N = A.shape[0]
    
    # Add a self loop
    A = A + np.identity(N)
    
    # Update the degree matrix (Because the added self-loops)
    D = np.diag(np.sum(A, axis=1))
    
    # Calculate A_hat and (D^{1/2})^{-1}
    D_sqrt_inv = scipy.linalg.inv(scipy.linalg.sqrtm(D))
    A_hat = D_sqrt_inv @ A @ D_sqrt_inv
    
    
    # Power iteration: A_tilde = (1-\alpha)(\sum_{i=0}^{K} \alpha^{i}\hat{A}^{i})
    A_tilde = np.zeros((N,N))
    A_hat_i = np.identity(N)
    alpha_i = 1
    for i in range(0, K+1):
        A_tilde = A_tilde + alpha_i*A_hat_i
        alpha_i = alpha_i*alpha
        A_hat_i = A_hat_i @ A_hat
    A_tilde = (1-alpha)*A_tilde
    
    # A_tilde: [N, N], 2-d float tensor
    return torch.tensor(A_tilde).type(torch.FloatTensor)

class cSBM:
    
    def __init__(self, N, p, d, mu, l):
        
        """
        N: number of nodes
        p: dimension of feature vector 
        d: average degree
        l: lambda, hyperparameter
        mu: mu, hyperparameter
        
        For details: https://arxiv.org/pdf/1807.09596.pdf
        and https://openreview.net/pdf/3fd51494885a4f0252dd144ae51025065fef2186.pdf
        """
        
        
        # Generate class from {-1, 1} for each node
        v = np.random.choice(a = [-1, 1],
                             size = N,
                             replace = True,
                             p = [0.5, 0.5])
        
        class1_ids = np.argwhere(v==1)
        
        class2_ids = np.argwhere(v==-1)
        
        
        # Mask -1 to 0 and store the result in v_mask
        v_mask = np.copy(v)
        v_mask[v==-1] = 0
        
        # calculate c_in and c_out
        c_in = d + np.sqrt(d)*l
        c_out = d - np.sqrt(d)*l
        
        
        # Generate a latent random vector u with size p
        u = np.random.normal(loc=0, scale=1/np.sqrt(p), size=p)
        
        # Generate the adjacent matrix without self-loop
        A = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1, N):
                if (v[i] == v[j]):
                    if (np.random.choice(a = [1,0], p = [c_in/N, 1-c_in/N])):
                        A[i,j] = 1.0
                    else:
                        A[i,j] = 0.0
                else:
                    if (np.random.choice(a = [1,0],p = [c_out/N, 1-c_out/N])):
                        A[i,j] = 1.0
                    else:
                        A[i,j] = 0.0
        A = A + A.T
        
        # Save all the necessary parameters
        self.v = v
        self.v_mask = v_mask
        self.A = A
        self.u = u
        self.p = p
        self.N = N
        self.mu = mu
        xi = N/p
        self.phi = np.arctan((l*np.sqrt(xi))/mu)*(2/np.pi)
        self.threshold = l**2 + (mu**2)/(N/p)
        self.class1_ids = class1_ids.reshape(-1)
        self.class2_ids = class2_ids.reshape(-1)
        self.Xs = None
        self.ys = None
       
    def generate_features(self, n_k):
        
        # Generate feature vectors for each node with size n_k.
        
        v = self.v
        p = self.p
        mu = self.mu
        u = self.u
        N = self.N
        
        Xs = []
        ys = []
        for i in range(N):
            X = []
            for j in range(n_k):
                x_j = np.sqrt(mu/N)*v[i]*u + np.random.normal(loc=0, scale=1, size=p)/np.sqrt(p)
                X.append(x_j)

            X = torch.tensor(np.array(X))
            
            if v[i] == -1:
                y = np.zeros(n_k)
            elif v[i] == 1:
                y = np.ones(n_k)

            y = torch.tensor(y).type(torch.LongTensor)
            Xs.append(X)
            ys.append(y)
        
        self.Xs = Xs
        self.ys = ys
          
            
        
        
def mean_agg(central_parameters, central_model, node_list, train_indices):
    
    num_train = len(train_indices)
    
    with torch.no_grad():
        
        for pname, param in central_model.named_parameters():
            
            p = node_list[train_indices[0]].model.state_dict()[pname]
            
            for i in range(1, num_train):
                
                p = p + node_list[train_indices[i]].model.state_dict()[pname]
            
            p = p/num_train
            
            central_parameters[pname] = p
            
            param.copy_(p)
import torch
import scipy
import copy
import scipy.linalg
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import mean_agg


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
    
    
class Node:
    
    
    def __init__(self, local_model, node_idx, X, y):
        
        """
        local model: The local MLP model for each node
        node_idx: The unique index of a node
        X: [n_k, p], feature matrix, float tensor
        y: [n_k], true labels, long tensor
        """
        
        self.model = local_model
        self.idx = node_idx
        self.X = X
        self.y = y
        self.n_k = X.shape[0]
        self.dataloader = None
        self.optimizer = None
        
        
    def upload_local_parameters(self):
        
        """
        Upload local model parameters to central server.
        Usually used for aggregation step in each communication.
        """
        
        return self.model.state_dict()
    
    
    def receieve_central_parameters(self, central_parameters):
        
        """
        central_parameters: A state dictonary for central server parameters.
        
        Receive the broadcasted central parameters.
        """
        
        with torch.no_grad():
            for pname, param in self.model.named_parameters():
                param.copy_(central_parameters[pname])
                
                
    def upload_h(self, gradient=True):
        
        
        """
        This function uploads an random hidden vector from a node to the central server.
        It also calculate and upload a dictonary of gradients  (dh/dw, 3D tensors) for each parameter w.r.t the local model
        """ 
        
        # x: [p]
        x = self.X[np.random.choice(a=self.n_k),:]
        
        if gradient:
            
            # Clear the possible accumulated gradient of the parameters of local model
            self.model.zero_grad()
        
            h = self.model(x).view(1, -1)
            
            num_class = h.shape[-1]

            dh = {}

            for i in range(num_class):

                h[0, i].backward(retain_graph=True)

                for pname, param in self.model.named_parameters():

                    if pname in dh:
                        dh[pname].append(param.grad.data.clone())
                    else:
                        dh[pname] = []
                        dh[pname].append(param.grad.data.clone())

                    if (i == num_class-1):
                        d1, d2 = dh[pname][0].shape
                        dh[pname] = torch.cat(dh[pname], dim=0).view(num_class, d1, d2)

                self.model.zero_grad()

            return h, dh
        
        else:
            with torch.no_grad():
                h = self.model(x).view(1, -1)
                
        return h
    
    
    def upload_data(self, m=1):
        
        if (m > self.n_k):
            raise ValueError("m is bigger than n_k!")
            
        
        ids = np.random.choice(a=self.n_k, size=m, replace=False)
        
        X = self.X[ids,:].view(m, 1, -1)
        
        y = self.y[ids].view(m, 1)
        
        return X, y
    
    def local_update(self, A_tilde_k, C_k, dH, I, 
                     opt="Adam",
                     learning_rate=0.01, num_epochs=10, 
                     gradient=True, gradient_clipping=None):
        
        """
        The local update process for a node k.
        
        A_tilde_k: The kth row of PageRank matrix A_tilde.
        
        C_k: [1, num_class] The aggregated neighborhood information for node k.
        
        dH: A list of gradient dictonaries, where the kth dictonary contains the gradients of each parameter for node k.
        
        I: Number of local updates.
        
        opt: Optimizer used for local updates: SGD or Adam. Default: "Adam"
        
        learning rate: learning rate for SGD. Default: 0.1
        
        gradient: boolean, whether to include the "fake gradient" or not. Default: True
        
        gradient_clipping: Whether to peform gradient clipping method during training process. None means no gradient clipping,
        if a number (int or float) is given, then the maximum norm is determined by this number. Default: None.
        """
        
        if (self.dataloader == None):
            batch_size = int(np.floor(self.n_k/I))
            dataset = torch.utils.data.TensorDataset(self.X, self.y)
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
        k = self.idx
        
        N = A_tilde_k.shape[0]
        
        num_class = C_k.shape[-1]
        
        if (opt == "Adam"):
            optimizer = optim.Adam(self.model.parameters())
            
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
            
        for epoch in range(num_epochs):

            for X_B, y_B in self.dataloader:

                optimizer.zero_grad()
                
                B = X_B.shape[0]
            
                H_B = self.model(X_B)
                y_B_onehot = torch.zeros(B, num_class)
                y_B_onehot[np.arange(B), y_B] = 1

                Z_B = A_tilde_k[k]*H_B + C_k
                y_B_hat = F.softmax(Z_B, dim=1)
                
                if (gradient == True and dH != None):
                    
                    batch_loss = F.nll_loss(torch.log(y_B_hat), y_B, reduction="sum")
                    batch_loss.backward()
                    
                    with torch.no_grad():
                        Errs = y_B_hat - y_B_onehot
                        for pname, param in self.model.named_parameters():
                            for i in range(N):
                                if (i != k):
                                    param.grad.data += A_tilde_k[i]*torch.tensordot(Errs, dH[i][pname], dims=1).sum(dim=0)
                            param.grad.data = param.grad.data/B
                            
                else:
                    batch_loss = F.nll_loss(torch.log(y_B_hat), y_B, reduction="mean")
                    batch_loss.backward()
                    
                            
                if (gradient_clipping == None):     
                    optimizer.step()
                    
                elif (type(gradient_clipping) == float or type(gradient_clipping) == int):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping, norm_type=2)
                    optimizer.step()
                else:
                    raise ValueError("Unkown type of gradient clipping value!")
                    
                
    def local_eval(self, central_model, A_tilde_kk, C_k):
        
        count = 0
        
        with torch.no_grad():
            H = central_model(self.X)
            y_hat = F.softmax(A_tilde_kk*H+C_k, dim=1)
            loss = F.nll_loss(torch.log(y_hat), self.y, reduction="mean")
            preds = torch.max(y_hat, dim=1)[1]
            count = (preds == self.y).sum().item()
            
        return loss.item(), count/self.n_k
    
    
    
class Central_Server:
    
    def __init__(self, node_list, A_tilde):
        
        """
        A_tilde: PageRank matrix
        node_list: A list contains objects from Node class
        """
        
        self.A_tilde = A_tilde
        self.node_list = node_list
        self.N = len(node_list)
        self.central_parameters = None
        self.cmodel = None
        
    def init_central_parameters(self, input_dim, hidden_dim, output_dim, nn_type):
        
        """
        Initialize the central server parameter dictonary
        """
        
        if (nn_type == "MLP"):
            self.cmodel = MLP(input_dim, hidden_dim, output_dim)
            
        elif (nn_type == "LR"):
            self.cmodel = LR(input_dim, output_dim)
            
        
        self.central_parameters = copy.deepcopy(self.cmodel.state_dict())
        
        
    def broadcast_central_parameters(self):
        
        """
        Broadcast the current central parameters to all nodes.
        Usually used after the aggregation in the end of each communication
        """
        
        if self.central_parameters == None:
            raise ValueError("Central parameters is None, Please initilalize it first.")
        
        for node in self.node_list:
            node.receieve_central_parameters(self.central_parameters)
        
    def collect_hs(self, gradient=True):
        
        """
        Collect h and dh from each node.
        """
        
        H = []
        
        if gradient:
            
            dH = []

            for i in range(self.N):
                h_i, dh_i = self.node_list[i].upload_h(gradient)
                H.append(h_i)
                dH.append(dh_i)

            # H: [N, num_class]
            H = torch.cat(H, dim=0)

            # dH: a list of gradient dictonaries
            return H, dH
        
        else:
            for i in range(self.N):
                h_i = self.node_list[i].upload_h(gradient)
                H.append(h_i)

            # H: [N, num_class]
            H = torch.cat(H, dim=0)
            
            return H, None
        
            
    def collect_data(self, m):
        
        Xs = []
        
        ys = []
        
        for node in self.node_list:
            
            X, y = node.upload_data(m)
            
            Xs.append(X)
            ys.append(y)
            
            
        # Xs; [m, N, p]
        # ys: [m, N]
            
        Xs = torch.cat(Xs, dim=1)
        
        ys = torch.cat(ys, dim=1)
        
        return Xs, ys
            
            
        
    def communication(self, train_indices, test_indices, I, 
                      aggregation=mean_agg, 
                      opt="Adam", learning_rate=0.1, 
                      num_epochs=10, gradient=True, m=10, 
                      gradient_clipping=None):
        
        """
        train_indices: A list of indices for the nodes that will be used during training.
        
        I: Number of local updates.
        
        test_indices: A list of indices for the nodes that will be used for testing purpose.
        
        num_epochs: Number of training epochs for each training node during local update.
        
        aggregation: aggregation method, for now, only mean aggregation is implemented. Default: mean_agg. 
        
        learning_rate: Learning rate for SGD. Default: 0.1
    
        opt: optimization method: Adam or SGD. Default: "Adam"

        gradient: boolean, whether to include the "fake gradient" or not. Default: True

        m: The number of feature vectors used for training loss evaluation in the end of each communication for each node. 
           Default: 10
           
        gradient_clipping: Whether to peform gradient clipping method during training process. None means no gradient clipping,
                           if a number (int or float) is given, then the maximum norm is determined by this number. 
                           Default: None.
        """
        
        self.broadcast_central_parameters()
        
        # H: [N, num_class]
        H, dH = self.collect_hs(gradient)
        
        # C: [N, num_class]
        with torch.no_grad():
            C = torch.matmul(self.A_tilde, H)
        
        for k in train_indices:
            with torch.no_grad():
                C_k = C[k,:] - self.A_tilde[k,k]*H[k,:]
    
            self.node_list[k].local_update(self.A_tilde[k,:], C_k, dH, 
                                           I, opt, learning_rate, num_epochs, gradient, gradient_clipping)
            
        aggregation(self.central_parameters, self.cmodel, self.node_list, train_indices)
        
        
        # Xs: [m, N, p]
        # ys: [m, N]
        Xs, ys = self.collect_data(m)
        
        with torch.no_grad():
            
            # Hs: [m, N, num_class]
            Hs = self.cmodel(Xs)
            
            # Zs: [m, N, num_class]
            Zs = torch.matmul(self.A_tilde, Hs)
            
            
            # train_Zs: [m, num_train, num_class]
            # train_ys: [m, num_train]
            train_Zs = Zs[:,train_indices,:]
            train_ys = ys[:,train_indices]
            
            num_train = len(train_indices)
            
            train_loss = F.cross_entropy(train_Zs.view(m*num_train, -1), train_ys.view(m*num_train)).item()
        
        return train_loss    
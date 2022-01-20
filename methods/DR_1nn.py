# This code is modified from protonet.py in https://github.com/wyharveychen/CloserLookFewShot

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

swc_update_rho=True 

device='cuda:0'

if not swc_update_rho:
    rho=10.0

class DR1NN(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(DR1NN, self).__init__( model_func,  n_way, n_support)        
        self.loss_fn = nn.CrossEntropyLoss() 
        
        if swc_update_rho:
            self.log_rho = torch.nn.Parameter(torch.tensor([2.0])) 
            self.log_rho.requires_grad = True
        
    
    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support = z_support.contiguous()
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 ) #shape: [n_way * n_query, dim]
        
        dists = euclidean_dist2(z_query, z_support) #shape: [n_way * n_query, n_way, n_support]
        min_dists=torch.min(dists,-1).values #shape: [n_way*n_query, n_way]

        if swc_update_rho:
            scores = -torch.exp(self.log_rho)*torch.log(min_dists)
        else:
            scores = -p*torch.log(min_dists)
        
        return scores
    
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )
 
    
def euclidean_dist2( x, y):
    # x: N x D
    # y: n_way x n_support x D
    N = x.size(0) #N = n_way * n_query
    n_way = y.size(0)
    n_support = y.size(1)
    D = x.size(1)
    
    assert D==y.size(2)

    x = x.unsqueeze(1).expand(N, n_way, D) #shape: [N, n_way, D]
    x = x.unsqueeze(2).expand(N, n_way, n_support, D) #shape: [N, n_way, n_support, D]
    y = y.unsqueeze(0).expand(N, n_way, n_support, D) #shape: [N, n_way, n_support, D]
     
    dists = torch.sqrt(torch.pow(x - y, 2).sum(-1)+1e-10) #Add small value to avoid getting Inf or NaN gradients values
    return dists

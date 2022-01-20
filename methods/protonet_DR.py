# This code is modified from protonet.py in https://github.com/wyharveychen/CloserLookFewShot

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

#Use DR-formulation

swc_update_rho=True 
device='cuda:0'

if not swc_update_rho:
    rho=10.0

class ProtoNet_DR(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet_DR, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        
        if swc_update_rho:
            self.log_rho = torch.nn.Parameter(torch.tensor([2.0])) 
            self.log_rho.requires_grad = True


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        if swc_update_rho:
            scores = -torch.exp(self.log_rho)*torch.log(dists)
        else:
            scores = -rho*torch.log(dists)
        return scores #−rho * ln(d_{x′,c})


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )
            


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.sqrt(torch.pow(x - y, 2).sum(2)+1e-10) #Note: it use distance (not squared distance)

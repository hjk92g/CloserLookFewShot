import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):       
        scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer ):
        print_freq = int(len(train_loader)/10)
        avg_loss=0

        for i, (x,_ ) in enumerate(train_loader):
            if i==len(train_loader)-1: #Get the positions of episode points (both support and query points) on embedding space just before the weight updates
                z_support1, z_query1  = self.parse_feature(x,False) #z_support1 shape: [n_way, n_support, dim], z_query1 shape: [n_way, n_query, dim]
                pre_z_all1 = torch.cat([z_support1, z_query1],dim=1)
                pre_z_all1 = pre_z_all1.view(-1,pre_z_all1.size(2)) #[n_way*(n_support+n_query), dim]
                z_all1 = pre_z_all1 -torch.mean(pre_z_all1,dim=-1,keepdim=True) #X_origin
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            
            optimizer.step()
            avg_loss = avg_loss+loss.item() #May need to replace "loss.item()" to "loss.data[0]" depending on the Pytorch version

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                
            if i==len(train_loader)-1: #Get the positions of episode points (both support and query points) on embedding space right after the weight updates
                z_support2, z_query2  = self.parse_feature(x,False)
                pre_z_all2 = torch.cat([z_support2, z_query2],dim=1)
                pre_z_all2 = pre_z_all2.view(-1,pre_z_all2.size(2)) #[n_way*(n_support+n_query), dim]
                z_all2 = pre_z_all2 -torch.mean(pre_z_all2,dim=-1,keepdim=True)  #X_new
                alpha, norm_ratio = self.analyze_scale_change(z_all1, z_all2) #alpha_hat* and norm ratio phi
                print('alpha: {:f} | norm_ratio: {:f}'.format(alpha, norm_ratio))
                converg, diverg, con_div_ratio = self.analyze_query_change(z_support1, z_query1, z_support2, z_query2) #psi_con, psi_div, and con_div_ratio psi_con/psi_div
                print('converg: {:f} | diverg: {:f} | con_div_ratio: {:f}'.format(converg, diverg, con_div_ratio))
                print('con_alpha_ratio: {:f} | div_alpha_ratio: {:f}'.format(converg/alpha, diverg/alpha))
                    

    def test_loop(self, test_loader, record = None, train_data=False):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support #x.size(1): batch_size for each class
            if self.change_way:
                self.n_way  = x.size(0) #x.size(0): number of ways
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100 )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        if train_data:
            print('%d Train Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        else:
            print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean
    
    def torch_norm(self, z):
        return torch.sum(z**2)**0.5

    def analyze_scale_change(self, z1, z2):
        #z shape: [n_way*(n_support+n_query), dim]
        #Conv4: dim=1600
        #ResNet18: dim=512
        
        alpha = torch.trace(torch.matmul(z1.t(),z2))/self.torch_norm(z1)**2 #alpha_hat* based on equation (11)
        norm_ratio=self.torch_norm(z2-alpha*z1)/self.torch_norm(z2-z1) #norm ratio phi
        return alpha, norm_ratio
    
    def analyze_query_change(self, support1, query1, support2, query2, mean=True):
        #support1 shape: [n_way, n_support, dim] 
        n_way = support1.size(0)
        n_support = support1.size(1)
        dim = support1.size(2)
        n_query = query1.size(1)
        
        support_exp1= support1.unsqueeze(2).unsqueeze(2) #[n_way, n_support, 1, 1, dim] 
        if mean:
            proto_exp1=torch.mean(support_exp1,axis=1,keepdim=True) #Mean: [n_way, 1, 1, 1, dim] 
        else:
            proto_exp1=support_exp1 #[n_way, n_support, 1, 1, dim] 
        query_exp1= query1.unsqueeze(0).unsqueeze(0) #[1, 1, n_way, n_query, dim] 
        dist1 = torch.sqrt(torch.pow(proto_exp1 - query_exp1, 2).sum(-1)) #[n_way, n_support, n_way, n_query] or [n_way, 1, n_way, n_query] 
        if not mean:
            dist1, min_ind1 = dist1.min(axis=1,keepdim=True) #[n_way, 1, n_way, n_query] #Use minimum (nearest) distance
            
        support_exp2= support2.unsqueeze(2).unsqueeze(2) #[n_way, n_support, 1, 1, dim] 
        if mean:
            proto_exp2= torch.mean(support_exp2,axis=1,keepdim=True) #Mean: [n_way, 1, 1, 1, dim] 
        else:
            proto_exp2=support_exp2 #[n_way, n_support, 1, 1, dim] 
        query_exp2= query2.unsqueeze(0).unsqueeze(0) #[1, 1, n_way, n_query, dim] 
        dist2 = torch.sqrt(torch.pow(proto_exp2 - query_exp2, 2).sum(-1)) #[n_way, n_support, n_way, n_query] or [n_way, 1, n_way, n_query]
        if not mean:
            dist2 = torch.gather(input=dist2,dim=1,index=min_ind1) #[n_way, 1, n_way, n_query]
        
        dist_ratio = dist2/dist1
        log_ratio_sm=[torch.log(dist_ratio[i,:,i,:]) for i in range(n_way)] #log ratio for same (correct) classes
        log_ratio_diff=[torch.log(dist_ratio[i,:,j,:]) for i in range(n_way) for j in range(n_way) if i!=j] #log ratio for different (incorrect) classes
        converg = torch.exp(torch.mean(torch.cat(log_ratio_sm))) #psi_con using geometric mean
        diverg = torch.exp(torch.mean(torch.cat(log_ratio_diff))) #psi_div using geometric mean
        con_div_ratio = converg/diverg
        return converg, diverg, con_div_ratio
        

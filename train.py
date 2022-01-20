import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet_S import ProtoNet_S
from methods.protonet_DR import ProtoNet_DR
from methods.softmax_1nn import Softmax1NN
from methods.DR_1nn import DR1NN
from io_utils import model_dict, parse_args, get_resume_file  

swc_update_rho=True 

def train(base_loader, val_loader, base_loader2, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')
        
    max_acc = 0       
    t1=time.time()
    for epoch in range(start_epoch,stop_epoch):
        loc_t1=time.time()
        model.train()
        if params.method in ['protonet_DR','DR_1nn']:
            if swc_update_rho:
                print('Epoch', epoch, '| rho',torch.exp(model.log_rho))
        
        model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        t2=time.time()
        print('Spend time (local training):',t2-loc_t1)
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        _ = model.test_loop( base_loader2, train_data=True) #Training accuracy (without data augmentation)
        acc = model.test_loop( val_loader) #Validation accuracy (without data augmentation)
        if acc > max_acc : 
            print("best model! save...") #Save best model based on validation accuracy
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar') #outfile: 
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        t2=time.time()
        print('Spend time (local training+validation):',t2-loc_t1)
        print('Spend time (total):',t2-t1)
        print()

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + 'base.json' 
    val_file   = configs.data_dir[params.dataset] + 'val.json' 
         
    if 'Conv' in params.model:
        image_size = 84 #Conv4
    else:
        image_size = 224 #ResNet18

    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.n_shot == 1:
            params.stop_epoch = 600
        elif params.n_shot == 5:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 600 #default
                
    print('method:',params.method)
    print('stop_epoch:',params.stop_epoch)
    print('n_support:',params.n_shot)
    print('n_way (train):',params.train_n_way)

    
    if params.method in ['protonet_S','protonet_DR','softmax_1nn','DR_1nn']: 
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        print('n_query (train):',n_query)

        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        base_loader2            = val_datamgr.get_data_loader( base_file, aug = False) #To report training accuracy without augmentation      

        if params.method == 'protonet_S':
            model           = ProtoNet_S( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'protonet_DR':
            model           = ProtoNet_DR( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'softmax_1nn':
            model           = Softmax1NN( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'DR_1nn':
            model           = DR1NN( model_dict[params.model], **train_few_shot_params )
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
        
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
        
    print(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
            
    print()
    model = train(base_loader, val_loader, base_loader2, model, optimization, start_epoch, stop_epoch, params)

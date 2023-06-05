import argparse, sys
import numpy as np
import torch
from utils.model import return_model, test, bkd_test, dba_test

sys.path.insert(0,'./utils')
from utils import *
from utils.adam import Adam
from utils.sgd import SGD
import torch.nn as nn
import torch.nn.parallel

import warnings
import time

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    use_cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser()
    torch.autograd.set_detect_anomaly(True)
    
    parser.add_argument(f"--config", default="grad", type=type("config"))
    args = parser.parse_args()
    config = yaml_config_hook("config/"+args.config+".yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.cuda.set_device(args.gpu)
    schedule=[1500]
    # dataset
    if args.agr in ["FLtrust"] or args.at_type in ["adtrust"] :
        if args.dataset in "CIFAR10":
            user_tr_data_tensors, user_tr_label_tensors,val_data_tensor,val_label_tensor,te_data_tensor,te_label_tensor, root_tr_data_tensors, root_tr_label_tensors = load_cifar10_fltrust(args)
            nusers=args.nusers
            user_tr_len=args.user_tr_len
            total_tr_len=user_tr_len*nusers
            val_len=args.val_len
            te_len=args.te_len
        elif args.dataset in ["FEMNIST"]:
            user_tr_data_tensors, user_tr_label_tensors,val_data_tensor,val_label_tensor,te_data_tensor,te_label_tensor, root_tr_data_tensors, root_tr_label_tensors = load_femnist_fltrust(args)
            nusers=args.nusers
            user_tr_len=args.user_tr_len
        elif args.dataset in ["MNIST"]:
            user_tr_data_tensors, user_tr_label_tensors,val_data_tensor,val_label_tensor,te_data_tensor,te_label_tensor, root_tr_data_tensors, root_tr_label_tensors= load_mnist_fltrust(args)
            nusers=args.nusers
            user_tr_len=args.user_tr_len
            total_tr_len=user_tr_len*nusers
            val_len=args.val_len
            te_len=args.te_len

    else:
        if args.dataset in "CIFAR10":
            user_tr_data_tensors, user_tr_label_tensors,val_data_tensor,val_label_tensor,te_data_tensor,te_label_tensor = load_cifar10(args)
            nusers=args.nusers
            user_tr_len=args.user_tr_len
            total_tr_len=user_tr_len*nusers
            val_len=args.val_len
            te_len=args.te_len
        elif args.dataset in ["FEMNIST"]:
            user_tr_data_tensors, user_tr_label_tensors,val_data_tensor,val_label_tensor,te_data_tensor,te_label_tensor = load_femnist(args)
            nusers=args.nusers
            user_tr_len=args.user_tr_len
        elif args.dataset in ["MNIST"]:
            user_tr_data_tensors, user_tr_label_tensors,val_data_tensor,val_label_tensor,te_data_tensor,te_label_tensor = load_mnist(args)
            nusers=args.nusers
            user_tr_len=args.user_tr_len
            total_tr_len=user_tr_len*nusers
            val_len=args.val_len
            te_len=args.te_len

    batch_size = args.batch_size
    nepochs = args.epochs
    nbatches = user_tr_len//batch_size
    opt = args.opt
    at_type = args.at_type
    print(at_type)
    criterion = nn.CrossEntropyLoss()
    n_attackers = args.n_attackers
    dev_type =  args.dev_type
    aggregation = args.agr
    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
    M = torch.Tensor()
    U = torch.Tensor()
    U_L = torch.Tensor()
    G = torch.Tensor()
    start_time = time.time()
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    
    for n_attacker in n_attackers:
        epoch_num = 0 
        best_global_acc = 0
        best_global_te_acc = 0
        at_fraction = n_attacker
        torch.cuda.empty_cache()
        r = np.arange(user_tr_len)
        r_m = np.arange(user_tr_len-1)
        r_c = np.arange(user_tr_len-2)

        if args.non_iid_p in [0.8, 0.9]:
            print("min1")
            r=r_m

        fed_model, _ = return_model(args.arch, 0.1, 0.9, parallel=False,device=args.gpu)
        #clustering model NONE
        model = None
        idx = None
        if "sgd" in opt:
            optimizer_fed = SGD(fed_model.parameters(), lr=args.fed_lr)
            if args.dataset in ["CIFAR10"]:
                optimizer_fed = SGD(fed_model.parameters(), lr=args.fed_lr,weight_decay=args.weight_decay,momentum=0.9)
        elif "adam" in opt:
            optimizer_fed = Adam(fed_model.parameters(), lr=args.fed_lr)
                
        while epoch_num <= nepochs:
            print("################# Epochs : ",epoch_num,"############")
            flag=1
            user_grads=[]

            if args.agr in ["FLtrust"] or args.at_type in ["adtrust"]:
                if args.dataset in ["FEMNIST"]:
                    chosen_users = np.random.choice(3400,nusers)
                    round_users = np.concatenate((chosen_users, [3400]), axis=0)
                    n_attacker = np.sum(round_users < (34*at_fraction))
                    print(f"round user : {len(round_users)} / at_fraction : {at_fraction} / n_attacker : {n_attacker}")
                    if args.at_type in ["dyn-krum","stat-krum"] or args.agr in ["bulyan"]:
                        n_attacker = max(4,n_attacker)
                        n_attacker = min(14,n_attacker)

                elif args.dataset in ["CIFAR10"]:
                    round_users = range(nusers+1)
                    if not epoch_num and epoch_num%nbatches == 0:
                        np.random.shuffle(r_c)
                        for i in range(nusers):
                            user_tr_data_tensors[i]=user_tr_data_tensors[i][r_c]
                            user_tr_label_tensors[i]=user_tr_label_tensors[i][r_c]

                elif args.dataset in ["MNIST"]:
                    round_users = range(nusers+1)
                    if not epoch_num and epoch_num%nbatches == 0:
                        np.random.shuffle(r_m)
                        for i in range(nusers):
                            user_tr_data_tensors[i]=user_tr_data_tensors[i][r_m]
                            user_tr_label_tensors[i]=user_tr_label_tensors[i][r_m]

            else:
                if args.dataset in ["FEMNIST"]:
                    round_users = np.random.choice(3400,nusers)
                    n_attacker = np.sum(round_users < (34*at_fraction))
                    if args.at_type in ["dyn-krum","stat-krum"] or args.agr in ["bulyan"]:
                        n_attacker = max(4,n_attacker)
                        n_attacker = min(14,n_attacker)

                elif args.dataset in ["CIFAR10"]:
                    round_users = range(nusers)
                    if not epoch_num and epoch_num%nbatches == 0:
                        np.random.shuffle(r)
                        for i in range(nusers):
                            user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                            user_tr_label_tensors[i]=user_tr_label_tensors[i][r]
                            
                elif args.dataset in ["MNIST"]:
                    round_users = range(nusers)
                    if not epoch_num and epoch_num%nbatches == 0:
                        np.random.shuffle(r)
                        for i in range(nusers):
                            user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                            user_tr_label_tensors[i]=user_tr_label_tensors[i][r]



            for i in round_users:
                if args.agr in ["FLtrust"]  or args.at_type in ["adtrust"]:
                    if args.dataset in ["FEMNIST"] :
                        if i==3400:
                            inputs = root_tr_data_tensors[0]
                            targets = root_tr_label_tensors[0]
                        else:                            
                            inputs = user_tr_data_tensors[i]
                            targets = user_tr_label_tensors[i]    
                                    
                    elif args.dataset in ["CIFAR10"]:
                        if i == nusers:
                            inputs = root_tr_data_tensors[0]
                            targets = root_tr_label_tensors[0]
                        else:
                            inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                            targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                    elif args.dataset in ["MNIST"]:
                        if i == nusers:
                            inputs = root_tr_data_tensors[0]
                            targets = root_tr_label_tensors[0]
                        else:
                            inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                            targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                else:
                    if args.dataset in ["FEMNIST"]:
                        inputs = user_tr_data_tensors[i]
                        targets = user_tr_label_tensors[i]                             
                    elif args.dataset in ["CIFAR10"]:
                        inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                        targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                    elif args.dataset in ["MNIST"]:
                        inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
                        targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]

                inputs, targets = inputs.cuda(), targets.cuda()
                
                if at_type == 'labelflip':
                    if i < 34*at_fraction and args.dataset in ["FEMNIST"]:
                        targets = args.n_classes - targets - 1
                    elif i < n_attacker:                
                        targets = args.n_classes - targets - 1
                
                if at_type == 'dyn-labelflip':
                    if i < 34*at_fraction and args.dataset in ["FEMNIST"]:
                        model_name= str(i) + ".pth"
                        p = str(args.non_iid_p)
                        PATH = os.path.join("./dlf/model", args.dataset, p , model_name)

                        surr_model, _ = return_model(args.arch, 0.1, 0.9, parallel=False,device=args.gpu)
                        surr_model.load_state_dict(torch.load(PATH))
                        pred = surr_model(inputs)
                        _, min_label = torch.min(pred,1)
                        targets = min_label       
                                         
                    elif i < n_attacker: 
                        model_name= str(i) + ".pth"
                        p = str(args.non_iid_p)
                        PATH = os.path.join("./dlf/model", args.dataset, p , model_name)

                        surr_model, _ = return_model(args.arch, 0.1, 0.9, parallel=False,device=args.gpu)
                        surr_model.load_state_dict(torch.load(PATH))
                        pred = surr_model(inputs)
                        _, min_label = torch.min(pred,1)
                        targets = min_label


                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = fed_model(inputs)

                '''
                if at_type == 'bkd':
                    if i < 34*at_fraction and args.dataset in ["FEMNIST"]:
                        loss =     bkd_loss(fed_model,inputs,targets,criterion,args) # scale
                    elif i < n_attacker:  
                        loss =     bkd_loss(fed_model,inputs,targets,criterion,args) # scale
                    else:
                        loss = criterion(outputs, targets)

                elif at_type == 'dba' :
                    if i < 34*at_fraction and args.dataset in ["FEMNIST"]:
                        loss = dba_loss(fed_model,inputs,targets,criterion,int(i / (34*(at_fraction/4))),args)# scale
                        
                    elif i < n_attacker and args.dataset in ["CIFAR"]:
                        if i in [0,1,2,3]:
                            loss =  dba_loss(fed_model,inputs,targets,criterion,0,args) # scale
                        elif i in [4,5,6,7]:
                            loss =  dba_loss(fed_model,inputs,targets,criterion,1,args) # scale
                        elif i in [8,9,10,11]:
                            loss =  dba_loss(fed_model,inputs,targets,criterion,2,args) # scale
                        elif i in [12,13,14,15]:
                            loss =  dba_loss(fed_model,inputs,targets,criterion,3,args) # scale
                            
                    elif i < n_attacker and args.dataset in ["MNIST"]:
                        loss =  dba_loss(fed_model,inputs,targets,criterion,int(i/(7)),args) # scale
                    else:
                        loss = criterion(outputs, targets)
                else:
                '''
                loss = criterion(outputs, targets)
                
                fed_model.zero_grad()
                
                loss.backward(retain_graph=True)
                
                param_grad=[]
                for param in fed_model.parameters():
                    if torch.any(torch.isnan(param.grad.data)):
                        assert(0)
                    '''
                    if at_type in ['bkd','dba'] and i < 34*at_fraction and args.dataset in ["FEMNIST"]:
                        param_grad=param.grad.data.view(-1)* nusers if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)*nusers))
                    elif at_type in ['bkd','dba'] and i < n_attacker:
                        param_grad=param.grad.data.view(-1)* nusers if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)*nusers)) 
                    '''
                    
                    param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))
                
                user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)
                
            if epoch_num in schedule and 0:
                for param_group in optimizer_fed.param_groups:
                    param_group['lr'] *= gamma

            #Generate malicious gradients
            if n_attacker > 0:

                if args.unknown:
                    sample_grads = user_grads[:n_attacker]
                    n_attacker_ = max(1, n_attacker**2//nusers)
                else:
                    sample_grads = user_grads[n_attacker:]
                    n_attacker_ = n_attacker

                if at_type == 'lie':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = lie_attack(sample_grads, agg_grads, z_values[12],dev_type)
                elif at_type == 'fang':                    
                    agg_grads = torch.mean(sample_grads, 0)
                    deviation = torch.sign(agg_grads)
                    mal_update = get_malicious_updates_fang(sample_grads, agg_grads, deviation, n_attacker_)
                elif at_type == 'our-median':                    
                    mal_update = our_attack_median(sample_grads, agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-max':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = our_attack_dist(sample_grads, agg_grads, n_attacker_, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = our_attack_score(sample_grads, agg_grads, n_attacker_,dev_type)
                elif at_type == 'our-dnc':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = our_attack_dnc(sample_grads, agg_grads, n_attacker_)
                elif at_type == 'new-dnc':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = new_attack_dnc(sample_grads, agg_grads, n_attacker_,fed_model,args)

                elif at_type == 'signflip':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = signflip(user_grads, agg_grads, n_attacker_)
                
                elif at_type == 'stat-mkrum':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = stat_attack_mkrum(sample_grads, agg_grads, n_attacker_,dev_type)
                elif at_type == 'stat-krum':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = stat_attack_krum(sample_grads, agg_grads, n_attacker_,dev_type)
                elif at_type == 'stat-bulyan':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = stat_attack_bulyan(sample_grads, agg_grads, n_attacker_,dev_type)
                elif at_type == 'stat-trmean':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = stat_attack_trmean(sample_grads,agg_grads,dev_type, n_attacker_)
                    
                elif at_type == 'stat-median':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = stat_attack_median(sample_grads,agg_grads,n_attacker_,dev_type)
                elif at_type == 'dyn-mkrum':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = dyn_attack_mkrum(sample_grads, agg_grads, n_attacker_,dev_type)
                elif at_type == 'dyn-krum':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = dyn_attack_krum(sample_grads, agg_grads, n_attacker_,dev_type)
                elif at_type == 'dyn-bulyan':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = dyn_attack_bulyan(sample_grads, agg_grads, n_attacker_,dev_type)
                elif at_type == 'dyn-trmean':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = dyn_attack_trmean(sample_grads,n_attacker_,dev_type)
                elif at_type == 'dyn-dnc':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = dyn_attack_dnc(sample_grads,agg_grads,n_attacker_,dev_type)
                elif at_type == 'dyn-median':
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = dyn_attack_median(sample_grads,n_attacker_,dev_type)


                elif at_type == 'adtrust':
                    global_grads = fltrust(user_grads)
                    agg_grads = torch.mean(sample_grads, 0)
                    mal_update = adtrust(user_grads,agg_grads,n_attacker_,global_grads,dev_type)                    
                    user_grads = user_grads[:-1]
                
                elif at_type == 'stat-adapt':

                    if epoch_num > 0:
                        agg_grads = torch.mean(sample_grads, 0)
                        mal_update = stat_attack_ours(sample_grads,agg_grads,n_attacker_,dev_type,rand_model,rand_reduced_idx,rand_I_std,rand_I_mean,lowv_model,lowv_reduced_idx,lowv_I_std,lowv_I_mean,test_grads,args)
                    else:
                        malicious_grads = user_grads
                        mal_updates = torch.Tensor().cuda()
                        mal_update = user_grads[:n_attacker]
                
                elif at_type == 'dyn-adapt':
                    if epoch_num > 0:
                        agg_grads = torch.mean(sample_grads, 0)
                        mal_update = dyn_attack_ours(sample_grads,agg_grads,n_attacker_,dev_type,rand_model,rand_reduced_idx,rand_I_std,rand_I_mean,lowv_model,lowv_reduced_idx,lowv_I_std,lowv_I_mean,test_grads,args)
                    else:
                        malicious_grads = user_grads
                        mal_updates = torch.Tensor().cuda()
                        mal_update = user_grads[:n_attacker]               
                elif at_type == 'stat-adapt':

                    if epoch_num > 0:
                        agg_grads = torch.mean(sample_grads, 0)
                        mal_update = stat_attack_ours(sample_grads,agg_grads,n_attacker_,dev_type,rand_model,rand_reduced_idx,rand_I_std,rand_I_mean,lowv_model,lowv_reduced_idx,lowv_I_std,lowv_I_mean,test_grads,args)
                    else:
                        malicious_grads = user_grads
                        mal_updates = torch.Tensor().cuda()
                        mal_update = user_grads[:n_attacker]
                
                elif at_type == 'dyn-adapt':
                    if epoch_num > 0:
                        agg_grads = torch.mean(sample_grads, 0)
                        mal_update = dyn_attack_ours(sample_grads,agg_grads,n_attacker_,dev_type,rand_model,rand_reduced_idx,rand_I_std,rand_I_mean,lowv_model,lowv_reduced_idx,lowv_I_std,lowv_I_mean,test_grads,args)
                    else:
                        malicious_grads = user_grads
                        mal_updates = torch.Tensor().cuda()
                        mal_update = user_grads[:n_attacker]
                        
                if at_type == 'no-atk':
                    malicious_grads = user_grads
                    mal_updates = torch.Tensor().cuda()
                elif at_type == "labelflip" or at_type== "bkd" or at_type == "dyn-labelflip" :
                    mal_updates = user_grads[:n_attacker]
                    malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)
                    user_grads = user_grads[n_attacker:]
                else:    

                    if len(mal_update.shape) < 2:
                        mal_updates = torch.stack([mal_update] * n_attacker)
                    else:
                        #mal_updates= mal_update
                        mal_updates = torch.stack([mal_update[0]] * n_attacker)
                    
                        
                    malicious_grads = torch.cat((mal_updates, user_grads[n_attacker:]), 0)
                    user_grads = user_grads[n_attacker:]
            else:
                mal_updates = torch.Tensor().cuda()
                malicious_grads = user_grads
                
            
            
        
            
            if args.dataset in "CIFAR10" and 0:
                if args.agr in ["flguard","flguard-fltrust", "flguard-mkrum", "flguard-signguard", "flguard-dnc", "flguard-median", "flguard-trmean", "flguard-bulyan", "flguard-fedavg"]:
                    if n_attacker == 0:     
                        U=torch.cat((U,user_grads.clone().cpu()), 0)
                    else:
                        M=torch.cat((M,mal_updates.clone().cpu()), 0)
                        U=torch.cat((U,user_grads.clone().cpu()), 0)
                
                    if epoch_num > args.budget:
                        if n_attacker == 0:
                            U = U[nusers:]
                        else:
                            M = M[n_attacker:]
                            U = U[nusers-n_attacker:]

            elif args.dataset in ["FEMNIST","MNIST", "CIFAR10"]:
                if args.agr in ["flguard", "flguard-fltrust", "flguard-mkrum", "flguard-signguard", "flguard-dnc", "flguard-median", "flguard-trmean", "flguard-bulyan", "flguard-fedavg"]:
                    
                    if n_attacker == 0 or at_type == 'no-atk':     
                        U=torch.cat((U,user_grads.clone().cpu()), 0)
                    else:
                        
                        U=torch.cat((U,mal_updates.clone().cpu()), 0)
                        U=torch.cat((U,user_grads.clone().cpu()), 0)
                        
                    
                    if epoch_num > args.budget:
                        U = U[nusers:]
                        U_L = U_L[nusers:]

            if args.agr in ["flguard-fltrust", "flguard-mkrum", "flguard-signguard", "flguard-dnc", "flguard-median", "flguard-trmean", "flguard-bulyan", "flguard-fedavg"]:
                
                if (epoch_num % args.budget == 0 ):
                    
                    if epoch_num == 0:
                        I = torch.cat((M,U),0)
                    else:
                        I = torch.cat((M,U),0)[:-nusers]

  
                    rand_model, rand_reduced_idx, rand_I_std, rand_I_mean = sub_train(I,args,dim_reduc ="random")
                    lowv_model, lowv_reduced_idx, lowv_I_std, lowv_I_mean = sub_train(I,args,dim_reduc ="low_var")
                
                if rand_model and lowv_model:
                    if at_type == 'no-atk':
                        test_grads = user_grads.clone()
                    else:
                        test_grads = torch.cat((mal_updates,user_grads),0)


            #aggregator for defense
            
            if args.agr in ["bulyan","flguard-bulyan"]  :
                if torch.any(torch.isnan(malicious_grads)):
                    exit(1)
                agg_grads, our_candidate=bulyan(malicious_grads, n_attacker)
                if at_type != 'no-atk':
                    tp +=  np.sum(our_candidate >= n_attacker)
                    fp += np.sum(our_candidate < n_attacker)
                    tn += (n_attacker - np.sum(our_candidate < n_attacker))
                    fn += (nusers - n_attacker -  np.sum(our_candidate >= n_attacker))
                else:
                    tp += len(our_candidate)
                    fn += nusers - len(our_candidate)
                if torch.any(torch.isnan(agg_grads)):
                    exit(1)
            elif args.agr in ["fedavg","flguard-fedavg"] :
                agg_grads = torch.mean(malicious_grads,dim=0)   
                             
            elif args.agr in ["dnc","flguard-dnc"]:
                agg_grads, our_candidate=dnc(malicious_grads,n_attacker)
                if at_type != 'no-atk':
                    tp +=  np.sum(our_candidate >= n_attacker)
                    fp += np.sum(our_candidate < n_attacker)
                    tn += (n_attacker - np.sum(our_candidate < n_attacker))
                    fn += (nusers - n_attacker -  np.sum(our_candidate >= n_attacker))
                else:
                    tp += len(our_candidate)
                    fn += nusers - len(our_candidate)

            elif args.agr in ["FLtrust","flguard-fltrust"] :
                agg_grads = fltrust(malicious_grads)

            elif args.agr in ["mkrum","flguard-mkrum"] :
                agg_grads, our_candidate=multi_krum(malicious_grads,n_attacker,multi_k=True)
                if at_type != 'no-atk':
                    tp +=  np.sum(our_candidate >= n_attacker)
                    fp += np.sum(our_candidate < n_attacker)
                    tn += (n_attacker - np.sum(our_candidate < n_attacker))
                    fn += (nusers - n_attacker -  np.sum(our_candidate >= n_attacker))
                else:
                    tp += len(our_candidate)
                    fn += nusers - len(our_candidate)

            elif args.agr in ["trmean","flguard-trmean"] :
                agg_grads = tr_mean(malicious_grads,n_attacker)

            elif args.agr in ["median", "flguard-median"] :
                agg_grads = median(malicious_grads,n_attacker)
                
            elif args.agr in ["signguard","flguard-signguard"] :
                if args.signguard_cluster in ["MeanShift"]:
                    agg_grads, our_candidate = signguard(malicious_grads, args.contra_n_cluster, args.signguard_cluster)
                else:
                    agg_grads, our_candidate = signguard(malicious_grads, args.contra_n_cluster)


                if at_type != 'no-atk':
                    tp +=  np.sum(our_candidate >= n_attacker)
                    fp += np.sum(our_candidate < n_attacker)
                    tn += (n_attacker - np.sum(our_candidate < n_attacker))
                    fn += (nusers - n_attacker -  np.sum(our_candidate >= n_attacker))
                else:
                    tp += len(our_candidate)
                    fn += nusers - len(our_candidate)

                if agg_grads is not None:
                    flag=1
                else:
                    flag=0
                    print("no update in this epoch:)__everything classified as mal")

            elif args.agr in ["pca"] :
                
                test_label,_ = pca_inference(model,malicious_grads,None,args)
                
                m_label = test_label[:n_attacker]
                u_label = test_label[n_attacker:]   

                if n_attacker > 0:
                    m_selected_cl = np.argmax(np.bincount(m_label))
                selected_cl = np.argmax(np.bincount(u_label))
     
                if n_attacker > 0:
                    m_selected_cl = np.argmax(np.bincount(m_label))

                if at_type != "no-atk":
                    selected_cl = np.argmax(np.bincount(test_label))
                    malicious_grads = torch.cat((mal_updates[m_label==selected_cl],user_grads[u_label==selected_cl]),0)                        
                else:
                    selected_cl = np.argmax(np.bincount(test_label))
                    malicious_grads = user_grads[selected_cl == test_label]
                        
                
                agg_grads = torch.mean(malicious_grads,dim=0) 



                # filtering stat
                if at_type != 'no-atk':
                    tp +=  len(user_grads[u_label==selected_cl])
                    fp += len(mal_updates[m_label==selected_cl])         
                    tn += len(mal_updates[m_label != selected_cl])
                    fn += len(user_grads[u_label!=selected_cl])
                else:
                    tp += len(malicious_grads)
                    fn += (nusers - len(malicious_grads))

                agg_grads = torch.mean(malicious_grads[:n_attacker],dim=0)

            elif args.agr in ["flguard"]:
                
                if (epoch_num % args.budget == 0 ):
                    
                    if epoch_num == 0:
                        I = torch.cat((M,U),0)
                    else:
                        I = torch.cat((M,U),0)[:-nusers]

  
                    rand_model, rand_reduced_idx, rand_I_std, rand_I_mean = sub_train(I,args,dim_reduc ="random")
                    lowv_model, lowv_reduced_idx, lowv_I_std, lowv_I_mean = sub_train(I,args,dim_reduc ="low_var")
                    
                
                if rand_model and lowv_model:
                    if at_type == 'no-atk':
                        test_grads = user_grads.clone()
                    else:
                        test_grads = torch.cat((mal_updates,user_grads),0)
                
                    rand_test_label, rand_z= sub_sc_inference(rand_model,test_grads,rand_reduced_idx,rand_I_std,rand_I_mean,args, n_attacker = n_attacker, n_cluster = args.contra_n_cluster)
                    lowv_test_label, z= sub_sc_inference(lowv_model,test_grads,lowv_reduced_idx,lowv_I_std,lowv_I_mean,args, n_attacker = n_attacker, n_cluster = args.contra_n_cluster)


                    #rand_m_label = rand_test_label[:n_attacker]
                    #rand_u_label = rand_test_label[n_attacker:]   

                    #lowv_m_label = lowv_test_label[:n_attacker]
                    #lowv_u_label = lowv_test_label[n_attacker:]   
                    if args.cluster in ["meanshift","Kmeans"]:
                        rand_unselected_cl = np.argmax(np.bincount(rand_test_label))
#print(rand_test_label)
                        rand_selected_idx = np.where(rand_test_label == rand_unselected_cl)
                        lowv_unselected_cl = np.argmax(np.bincount(lowv_test_label))
#                        print(lowv_test_label)
                        lowv_selected_idx = np.where(lowv_test_label == lowv_unselected_cl)

                        our_candidate = np.intersect1d(rand_selected_idx,lowv_selected_idx)
                    
                    elif args.contra_n_cluster > 2:
                        print("n_cluster:", args.contra_n_cluster)
                        rand_unselected_cl = np.argmax(np.bincount(rand_test_label))
                        rand_selected_idx = np.where(rand_test_label == rand_unselected_cl)
                        lowv_unselected_cl = np.argmax(np.bincount(lowv_test_label))
                        lowv_selected_idx = np.where(lowv_test_label == lowv_unselected_cl)

                        our_candidate = np.intersect1d(rand_selected_idx,lowv_selected_idx)

                    elif at_fraction > 50:
                        print("at_fraction:", at_fraction)
                        rand_unselected_cl = np.argmax(np.bincount(rand_test_label))
                        rand_selected_idx = np.where(rand_test_label != rand_unselected_cl)
                        lowv_unselected_cl = np.argmax(np.bincount(lowv_test_label))
                        lowv_selected_idx = np.where(lowv_test_label != lowv_unselected_cl)

                        our_candidate = np.intersect1d(rand_selected_idx,lowv_selected_idx)
                        
                    else:
                        rand_unselected_cl = np.argmin(np.bincount(rand_test_label))
                        rand_selected_idx = np.where(rand_test_label != rand_unselected_cl)
                        lowv_unselected_cl = np.argmin(np.bincount(lowv_test_label))
                        lowv_selected_idx = np.where(lowv_test_label != lowv_unselected_cl)
                        our_candidate = np.intersect1d(rand_selected_idx,lowv_selected_idx)
                        
                        
                    
                    if at_type != 'no-atk':
                        tp +=  np.sum(our_candidate >= n_attacker)
                        fp += np.sum(our_candidate < n_attacker)
                        tn += (n_attacker - np.sum(our_candidate < n_attacker))
                        fn += (nusers - n_attacker -  np.sum(our_candidate >= n_attacker))
                    else:
                        tp += len(our_candidate)
                        fn += nusers - len(our_candidate) 
                    if len(our_candidate) < 1:
                        flag == 0
                        print("no update in this epoch:)__everything classified as mal")
#	agg_grads = torch.mean(malicious_grads,dim=0) 
                    else:
                        agg_grads = torch.mean(malicious_grads[our_candidate],dim=0) 
                    
                    if torch.any(torch.isnan(agg_grads)):
                        assert(0)
                
                
            del user_grads

            start_idx=0
            
            optimizer_fed.zero_grad()

            model_grads=[]
            if flag == 1:
                for i, param in enumerate(fed_model.parameters()):
                    param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                    if torch.any(torch.isnan(param_)):
                        exit(1)
                    start_idx=start_idx+len(param.data.view(-1))
                    param_=param_.cuda()
                    model_grads.append(param_)
                    

                optimizer_fed.step(model_grads)

                
                te_loss, te_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
                val_loss = te_loss
                val_acc = te_acc
                
                is_best = best_global_acc < val_acc

                best_global_acc = max(best_global_acc, val_acc)

                if is_best:
                    best_global_te_acc = te_acc


                print('%s: at %s(%s) n_at %d n_mal_sel %d e %d val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type,dev_type, n_attacker, 1, epoch_num, val_loss, val_acc, best_global_acc, best_global_te_acc))                
                print(f"ASR {te_acc:.4f}")
                
                print(f"tp tn fp fn {tp} {tn} {fp} {fn}")

            epoch_num+=1

    

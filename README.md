# FLGuard: Byzantine-Robust Federated Learning via Ensemble of Contrastive Models



This code is based on virat shejwalker code(https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning). We have modified the code to the form of python code. We add SOTA agrs(fltrust, FLGuard(ours))  and a SOTA attack (fltrust adaptive attack). All configure file we experiment are given under config directory. Available options are below

- We have given codes for 10 aggregation algorithms in utils/defences.py : FedAVG(base), Trim-mean, Median, Krum, Multi-Krum, Bulyan, DnC, FLTrust, SignGuard, FLGuard.
- We have given codes for 7 model poisoning attack algorithms in utils/attacks.py : LIE, MinMax, MinSum, STAT-OPT, DYN-OPT, FLTrust-Adaptive attack, Our-Adaptive attack.
- We use three datasets (MNIST, CIFAR-10, FEMNIST). 

## Requirements
the necessary python packages are in the env.yml 

## Getting Started 
It's super easy to get our codes up and running
```bash
python train_fl.py --config [config_file_path]

```

For example, 
```bash
python train_fl.py --config bulyan/cifar/1/res_adtrucontra_bulyan_cifar  # prefix path : config/
python train_fl.py --config adaptive_with_defense/cifar/1/res_dyn_adapt_trmean_cifar_ahc  # prefix path : config/
```

## Configuration Details

You can modify the experiment settings with configuration file (examples are under config/)

1. General Settings 
```bash
seed: 42 # random seed
workers: 32 # GPU workers
dataset_dir: "./datasets" # datasets prefix path 
gpu: 0 # GPU
```

2. Federated Settings 
```bash
###  Federated Settings  ###
arch: "resnet-14" # Image Classification Model (For FL)
# [ MNIST : mnist-fcn / FEMNIST : mnist-conv / CIFAR-10 : resnet-14 ]

dataset: "CIFAR10" # MNIST / FEMNIST / CIFAR-10
n_classes: 10 # the number of classes
non_iid_p: 0.0 # the non-iid degree of dataset for CIFAR10, MNIST
nusers: 50 # the number of FL clients
user_tr_len: 1000 # Dataset size for each client
val_len: 5000
te_len : 5000

batch_size: 32 # FL batch size
epochs: 2000 # the total number of FL epochs

fed_lr: 0.01 # FL model learning rate
weight_decay: 0.0005 # FL model weight decay for CIFAR10
opt: 'sgd' # FL optimizer : [ sgd / adam ]

agr: "st4" # FL aggregation algorithms : [ fedavg / trmean / median / krum / mkrum / dnc/ fltrust / signguard/ st4 ]
# st4 is our FLGuard
```

3. Adversary Settings
```bash
###  Adversary Settings ###
at_type: "lie" # attack method : [ lie / min-max / min-sum / stat-krum / stat-trmean / stat-median /  dyn-krum / dyn-trmean / dyn-median / dyn-dnc / adtrust / labelflip]
dev_type: "std" # perturbation type
unknown: False # For adversaries' capability type2, 4, This set to TRUE.
n_attackers: # For CIFAR-10 and MNIST, the number of malicious clients. For FEMNIST, the percentage of malicious clients
 - 20
 
```
4. Contrastive learning Settings
```bash
### Contrastive Learning Settings ###
contra_model_mode: "ae"  # contrastive learning model : [ ae / ResNet18 ]
dim_reduction : "low_var" # reduction method for NOT ensemble model 

# ae is a simple FCN model
contra_dim0: 3072
contra_dim1: 3072
contra_dim2: 3072
contra_n_cluster: 2
# FCN dimensions

# augmentation options 
contra_n_subsets: 2
contra_overlap: 1.0

# learning settings
contra_normalize: true
contra_p_norm: 2
contra_reconstruct_subset: False

contra_epochs: 5 # the total number of contrastive learning epochs
contra_dropout: 0.2 # contrastive learning dropout
contra_learning_rate: 0.001 # CL learning rate
contra_batch_size: 32 # CL batch size

contra_add_noise: True # it must be set to TRUE
contra_noise_type: "gaussian_noise" # CL augmentation type
contra_masking_ratio: 1 # CL augmentation ratio
contra_noise_level: 0.0001 # CL augmentation level

contra_contrastive_loss: True # Must set to True
contra_reconstruction: True # Must set to True

contra_shallow_architecture: True # Must set to True
contra_similarity: "l2" # the similarity metric for CL : [ l2 / dot / cosine_similarity ]
contra_tau: 0.5 # CL tau parameter

contra_isBatchNorm: false 
contra_isDropout: True 

contra_validate: False    
contra_scheduler: False    

budget: 5 # Ours FL round budget size
cluster: ahclu # clustering method
```


## Project File Tree 
```bash
├── config
├── datasets
│   ├── cifar-10-batches-py
│   └── MNIST
│       └── raw
├── env.yml
├── modules
│   ├── aewrapper.py
│   └── contrastive_module.py
├── test.py
├── train_fl.py
└── utils
    ├── __init__.py
    ├── adam.py    
    ├── attacks.py
    ├── defences.py
    ├── load_data.py
    ├── loss_func.py
    ├── model.py
    ├── models
    │   ├── cifar
    │   │   ├── alexnet.py
    │   │   ├── convnet.py
    │   │   ├── densenet.py
    │   │   ├── __init__.py
    │   │   ├── preresnet.py
    │   │   ├── resnet.py
    │   │   ├── resnext.py
    │   │   ├── vgg.py
    │   │   └── wrn.py
    │   ├── imagenet
    │   │   ├── __init__.py
    │   │   └── resnext.py
    │   └── __init__.py
    ├── sgd.py
    └── yaml_config_hook.py



```


## Reference 

```
@inproceedings{shejwalkar2021manipulating,
  title={Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning},
  author={Shejwalkar, Virat and Houmansadr, Amir},
  booktitle={NDSS},
  year={2021}
}

@article{cao2021fltrust,
  title={Fltrust: Byzantine-robust federated learning via trust bootstrapping},
  author={Cao, Xiaoyu and Fang, Minghong and Liu, Jia and Gong, Neil Zhenqiang},
  booktitle={NDSS},
  year={2021}
}
```










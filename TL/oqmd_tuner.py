import torch
from matdeeplearn import models, process, training
from torch_geometric.data import DataLoader, Dataset
from matdeeplearn.training.training import evaluate
from matdeeplearn.models.utils import model_summary
import pickle
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import cook_initial_point_generator
import argparse, sys
import numpy as np

config = {'Job': {'hyper_concurrency': 8,
         'hyper_delete_processed': 'True',
         'hyper_iter': 1,
         'hyper_resume': 'True',
         'hyper_trials': 50,
         'hyper_verbosity': 1,
         'job_name': 'tl_megnet',
         'model': 'MEGNet_demo',
         'reprocess': 'False',
         'seed': 42},
 'Models': {'act': 'relu',
            'batch_norm': 'True',
            'batch_size': 50,
            'batch_track_stats': 'True',
            'dim1': 100,
            'dim2': 100,
            'dim3': 100,
            'dropout_rate': 0.4,
            'epochs': 500,
            'gc_count': 4,
            'gc_fc_count': 1,
            'lr': 0.0005,
            'model': 'MEGNet',
            'optimizer': 'AdamW',
            'optimizer_args': {},
            'pool': 'global_mean_pool',
            'pool_order': 'early',
            'post_fc_count': 3,
            'pre_fc_count': 1,
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_args': {'factor': 0.8,
                               'min_lr': 1e-05,
                               'mode': 'min',
                               'patience': 10,
                               'threshold': 0.0002}},
 'Processing': {'SM_descriptor': 'False',
                'SOAP_descriptor': 'False',
                'SOAP_lmax': 4,
                'SOAP_nmax': 6,
                'SOAP_rcut': 8.0,
                'SOAP_sigma': 0.3,
                'data_format': 'cif',
                'data_path': 'data/hse/initial/',
                'dataset_type': 'inmemory',
                'dictionary_path': 'atom_dict.json',
                'dictionary_source': 'default',
                'edge_features': 'True',
                'graph_edge_length': 50,
                'graph_max_neighbors': 12,
                'graph_max_radius': 8.0,
                'target_path': 'targets.csv',
                'verbose': 'True',
                'voronoi': 'False'},
 'Training': {'loss': 'l1_loss',
              'target_index': 0,
              'test_ratio': 0.1,
              'train_ratio': 0.8,
              'val_ratio': 0.1,
              'verbosity': 5}}

config["Job"]["run_mode"] = 'Training'
config["Job"]["save_model"] = "True"
config["Job"]["model_path"] = 'tlmodel2.pt'
config["Job"]["write_output"] = "True"

# Parse command line arguments
parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
parser.add_argument("--filename",default='tmp_model.pth',type=str,help="name used for storing models temporarily")
parser.add_argument('--output',default='output.pkl',type=str,help="name used for storing bayesian optimization search results on hyperparameters")
parser.add_argument('--n-initial-points',default=20,type=int,help="number of initial points sampled before further sampling with acqusition function")
parser.add_argument('--n-calls',default=50,type=int,help="Total number of times models are trained")
parser.add_argument('--seed',default=42,type=int,help="Random seed used for the bayesian optimizer")
parser.add_argument('--data-path',default='data/hse/initial/',type=str,help="Location of stored cif files with the targets.csv file")
parser.add_argument('--checkpoint', default='checkpoints/ihse.checkpoint',type=str,help="Location where weights of the best model are stored")
parser.add_argument('--pretrained', default=0, help='0 for model with 7 gnn layers; 1 for model with 6 gnn layers', type=int)
parser.add_argument('--jobname', default='tl_megnet', type=str, help='Prefix for record keeping')
parser.add_argument('--loss', default='l1_loss', type=str, help='Loss function to be used for training')
parser.add_argument('--epochs', default=50, type=int, help='epochs to train the model for')
parser.add_argument('--classification', default=0, type=int, help='0 for regression, 1 for classification')
args = parser.parse_args(sys.argv[1:])

config['Processing']['data_path'] = args.data_path
config['Job']['job_name'] = args.jobname
config['Training']['loss'] = args.loss
config['Models']['epochs'] = args.epochs

# Load the dataset and saved weights
dataset = process.get_dataset(config["Processing"]["data_path"], config["Training"]["target_index"], config["Job"]["reprocess"], config["Processing"])
rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.pretrained == 0:
    saved = torch.load('checkpoints/DEFAULT_ad8605a0_35_batch_size=4000,dim1=70,dim2=190,dim3=110,gnn_count=7,lr=0.00076326,pool=global_max_pool,post_fc_count=2,hyper_2022-01-17_17-33-58.checkpoint',map_location=rank)
elif args.pretrained == 1:
    saved = torch.load('checkpoints/DEFAULT_b3dc47f0_20_batch_size=3000,dim1=190,dim2=180,dim3=90,gnn_count=6,lr=0.0080614,pool=set2set,post_fc_count=5,hyper_concurre_2022-01-14_17-38-55.checkpoint',map_location=rank)

# Define the region of interest
dim1 = Real(1e-6, 0.09, prior='log-uniform') # learning-rate
dim2 = Integer(16,2048) # batch-sizes
dim4 = Real(0.0, 0.9) # dropout
if args.pretrained == 1:
    dim3 = Integer(0,5) # Number of fully connected post convolution dense layers to modify exlucding the last layer
else:
    dim3 = Integer(0,2)
dimensions = [dim2, dim1, dim3, dim4]

minimum_loss_so_far = np.infty

def tuning(input_params):
    print("input params: ", input_params) 
    # Modify currnent learning rate and batch size
    config["Models"]["batch_size"] = int(input_params[0])
    config["Models"]["lr"] = input_params[1]
    config["Models"]["dropout_rate"] = input_params[3]

    # Define architecture
    if args.pretrained == 0:
        model = models.MEGNet(dataset,dim1=70,dim2=190,dim3=110,gnn_count=7,lr=0.00027267,pool='global_max_pool',post_fc_count=2,pre_fc_count=1, gc_count=4, gc_fc_count = 1, dropout_rate=input_params[3])
    elif args.pretrained == 1:
        model = models.MEGNet(dataset,dim1=190,dim2=180,dim3=90,gnn_count=6,lr=0.0080614,pool='set2set',post_fc_count=5,pre_fc_count=1, gc_count=4, gc_fc_count = 1, dropout_rate=input_params[3])
    # Load pre-trained model weights
    model.load_state_dict(saved[0])
    
    # Freezing weights of convolution layers
    for idx,child in enumerate(model.children()):
        if idx==6:
            for sub_idx, sub_child in enumerate(child.children()):
                if sub_idx < len(list(child.children())) - input_params[2]:
                    for param in sub_child.parameters():
                        param.requires_grad = False

                else:
                        sub_child.reset_parameters()

        elif idx not in [7, 8, 9]:
            for param in child.parameters():
                param.requires_grad = False
       
        else:
            child.reset_parameters()

    #print(model_summary(model))
    world_size = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    error_vals = training.train_regular(device, world_size, config["Processing"]["data_path"], config["Job"], config["Training"], config["Models"], model, args.filename, args.classification==1)
    print("Params: ", input_params, "; MAE/ f1-score: ", error_vals[1])

    torch.save(model.state_dict(), args.checkpoint + str(input_params[0]) + str(input_params[1]) + str(input_params[2]) + str(input_params[3])  )

    if args.classification:
        return -1*error_vals[1]

    return error_vals[1]

# Define the sampling strategy for initial point generation and define the bayesian optimizer object
lhs_max = cook_initial_point_generator("lhs", criterion="maximin")
result = gp_minimize(func=tuning, dimensions=dimensions, n_calls=args.n_calls, n_initial_points=args.n_initial_points, random_state=args.seed, initial_point_generator=lhs_max)

# Write the output of analysis
with open(args.output, 'wb') as f:
    pickle.dump(result, f)

print("Best parameters: ", result.x)
print("Best performance: ",result.fun)


from utils import *
from models import *
from training import *

import argparse

parser = argparse.ArgumentParser(description="Define the type of training")
parser.add_argument("-lr", "--learning_rate", help="initial learning rate",
			default=0.05,
			type=float)
parser.add_argument("-b", "--batch_size", help="batchsize",
			default=32,
			type=int)
parser.add_argument("-d", "--depthwise", help="depthwise conv or not",
                    action='store_true')

args = parser.parse_args()

hpc = False
labs = False
paperspace = True

if hpc:
    main_data_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/model_data"
    main_logs_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/predicted_streamlines"
elif labs:
    main_data_path = "/media/hsa22/Expansion/Brain_MINDS/model_data"
    main_logs_path = "/media/hsa22/Expansion//Brain_MINDS/predicted_streamlines"
elif paperspace:
    main_data_path = "/storage/model_data_w_resize"
    main_logs_path = "/notebooks/predicted_streamlines"
else:
    main_data_path = "D:\\Brain-MINDS\\model_data"
    main_logs_path = "D:\\Brain-MINDS\\predicted_streamlines"

# Define the main name
pred_name = "autoenc"
train_name = "training_logs"

# Parse arguments
init_lr = args.learning_rate
batch_size = args.batch_size
depthwise = args.depthwise

print("depthwise is", depthwise)

# Define the paths
autoenc_arrays_path = os.path.join(main_logs_path, "autoenc_predictions", pred_name)
training_log_folder = os.path.join(main_logs_path, train_name)
model_folder = os.path.join(main_logs_path, "models", pred_name)

check_output_folders(autoenc_arrays_path, "autoenc arrays", wipe=False)
check_output_folders(training_log_folder, "training_log_folder", wipe=False)
check_output_folders(model_folder, "model_folder", wipe=False)

training_log_path = os.path.join(training_log_folder, "autoencoder.csv")
model_filename = os.path.join(model_folder, "autoencoder.h5")

# Create the configs dictionary
configs = {

    ####### Model #######
    "model_name" : "autoencoder", # Model name
    "input_nc" : 1,
    "combination" : True, # Combination
    "task" : "autoencoder", # Task
    "depthwise_conv" : depthwise, # Depthwise convolution
    "hidden_size" : 100, # number of neurons
    "library_opt" : True, # Use stuff from torch_optim
    "previous" : True, # Whether or not to include previous predictions
    
    ####### Training #######
    "n_epochs" : 50, # Number of epochs
    "loss" : "MSE_loss", # Loss function
    "optimizer" : "Adam", # Optimizer
    "evaluation_metric" : "MSE_loss", # Evaluation metric
    "shuffle_dataset" : True,
    "separate_hemisphere" : False,
    "cube_size" : 5, # cube size
    "save_best" : True, # Save best model
    "overfitting" : False, # Overfitting

    ####### Data #######
    "main_data_path" : main_data_path, # Data path
    "training_log_path" : training_log_path, # Training log path
    "model_filename" : model_filename, # Model filename
    "autoenc_arrays_path" : autoenc_arrays_path, # Path to the streamlines array
    "batch_size" : batch_size, # Batch size
    "validation_batch_size" : batch_size, # Validation batch size
    "num_streamlines" : 200, # Number of streamlines to consider from each site
    
    ####### Parameters #######
    "initial_learning_rate" : init_lr, # Initial learning rate
    "early_stopping_patience": 15, # Early stopping patience
    "decay_patience": 2, # Learning rate decay patience
    "decay_factor": 0.6, # Learning rate decay factor
    "min_learning_rate": 1e-08, # Minimum learning rate
    "save_last_n_models": 10, # Save last n models

    ####### Misc #######
    "skip_val" : False, # Skip validation
    "training_type" : "autoencoder", # Training type
    "tck_type" : "trk" # TCK type

}

# Define the configuration path and save it as a .json file
config_path = os.path.join("configs", configs["model_name"] + ".json")

# Save the configuration
dump_json(configs, config_path)

# Load the configuration
configs = load_json(config_path)

# Define the metric to monitor based on whether we're skipping val or not
if configs["skip_val"]:
    metric_to_monitor = "val_loss"
else:
    metric_to_monitor = "train_loss"

# Define the groups
if configs["skip_val"]:
    groups = ("training",)
else:
    groups = ("training", "validation")

model_metrics = (configs["evaluation_metric"],)

run_training(configs, metric_to_monitor=metric_to_monitor, bias=None)
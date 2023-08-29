from utils import *
from models import *
from training import *

import argparse

parser = argparse.ArgumentParser(description="Define the type of training")
parser.add_argument("-t", "--task", help="what task to run", 
			default="classification", required=True,
			type=str)
parser.add_argument("-c", "--contrastive", help="what type of contrastive to run if running contrastive",
			default="",
			type=str)
parser.add_argument("-d", "--depthwise", help="depthwise conv or not",
                    action='store_true')
parser.add_argument("-lr", "--learning_rate", help="initial learning rate",
			default=0.05,
			type=float)
parser.add_argument("-b", "--batch_size", help="batchsize",
			default=32,
			type=int)

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
pred_name = "baseline_mlp"
train_name = "training_logs"

# Parse arguments
task = args.task
contrastive = args.contrastive
depthwise = args.depthwise
init_lr = args.learning_rate
batch_size = args.batch_size

print("depthwise is", depthwise)

# If contrastive is "", set to False
if contrastive == "":
    contrastive = False

# Append task and contrastive to name
pred_name = pred_name + "_" + task
train_name = train_name + "_" + task
if contrastive != False:
    pred_name = pred_name + "_" + contrastive
    train_name = train_name + "_" + contrastive

streamline_arrays_path = os.path.join(main_logs_path, "streamline_predictions", pred_name)
training_log_folder = os.path.join(main_logs_path, train_name)
model_folder = os.path.join(main_logs_path, "models", pred_name)

check_output_folders(streamline_arrays_path, "streamline arrays", wipe=False)
check_output_folders(training_log_folder, "training_log_folder", wipe=False)
check_output_folders(model_folder, "model_folder", wipe=False)

training_log_path = os.path.join(training_log_folder, "baseline_mlp_streamlines.csv")
model_filename = os.path.join(model_folder, "baseline_mlp_streamlines.h5")

# Create the configs dictionary
configs = {

    ####### Model #######
    "model_name" : "baseline_mlp", # Model name
    "input_nc" : 45,
    "combination" : True, # Combination
    "task" : task, # Task
    "hidden_size" : 100, # number of neurons
    "depthwise_conv" : depthwise, # Depthwise convolution
    "library_opt" : True, # Use stuff from torch_optim
    "contrastive" : contrastive, # Contrastive
    "previous" : True, # Whether or not to include previous predictions
    
    ####### Training #######
    "n_epochs" : 50, # Number of epochs
    "loss" : "negative_log_likelihood_loss", # Loss function
    "optimizer" : "Adam", # Optimizer
    "evaluation_metric" : "negative_log_likelihood_loss", # Evaluation metric
    "shuffle_dataset" : True,
    "separate_hemisphere" : False,
    "cube_size" : 5, # cube size
    "save_best" : True, # Save best model
    "overfitting" : False, # Overfitting

    ####### Data #######
    "main_data_path" : main_data_path, # Data path
    "training_log_path" : training_log_path, # Training log path
    "model_filename" : model_filename, # Model filename
    "streamline_arrays_path" : streamline_arrays_path, # Path to the streamlines array
    "batch_size" : batch_size, # Batch size
    "validation_batch_size" : batch_size, # Validation batch size
    "num_streamlines" : -1, # Number of streamlines to consider from each site
    
    ####### Parameters #######
    "initial_learning_rate" : init_lr, # Initial learning rate
    "early_stopping_patience": 50, # Early stopping patience
    "decay_patience": 20, # Learning rate decay patience
    "decay_factor": 0.5, # Learning rate decay factor
    "min_learning_rate": 1e-08, # Minimum learning rate
    "save_last_n_models": 10, # Save last n models

    ####### Misc #######
    "skip_val" : False, # Skip validation
    "training_type" : "streamline", # Training type
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
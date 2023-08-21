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
pred_name = "resnet"
test_name = "testing_logs"

# Parse arguments
task = args.task
contrastive = args.contrastive
depthwise = args.depthwise

print("depthwise is", depthwise)

# If contrastive is "", set to False
if contrastive == "":
    contrastive = False

# Append task and contrastive to name
pred_name = pred_name + "_" + task
test_name = test_name + "_" + task
if contrastive != False:
    pred_name = pred_name + "_" + contrastive
    test_name = test_name + "_" + contrastive

streamline_arrays_path = os.path.join(main_logs_path, "streamline_predictions", pred_name)
testing_log_folder = os.path.join(main_logs_path, test_name)
model_folder = os.path.join(main_logs_path, "models", pred_name)

check_output_folders(streamline_arrays_path, "streamline arrays", wipe=False)
check_output_folders(testing_log_folder, "testing_log_folder", wipe=False)
check_output_folders(model_folder, "model_folder", wipe=False)

testing_log_path = os.path.join(testing_log_folder, "resnet_streamlines.csv")
model_filename = os.path.join(model_folder, "resnet_streamlines.h5")

# Create the configs dictionary
configs = {

    ####### Model #######
    "model_name" : "resnet_streamlines", # Model name
    "input_nc" : 1,
    "combination" : True, # Combination
    "task" : task, # Task
    "hidden_size" : 100, # number of neurons
    "depthwise_conv" : depthwise, # Depthwise convolution
    "library_opt" : True, # Use stuff from torch_optim
    "contrastive" : contrastive, # Contrastive
    "previous" : True, # Whether or not to include previous predictions
    
    ####### Training #######
    "loss" : "negative_log_likelihood_loss", # Loss function
    "evaluation_metric" : "negative_log_likelihood_loss", # Evaluation metric
    "separate_hemisphere" : False,
    "cube_size" : 5, # cube size

    ####### Data #######
    "main_data_path" : main_data_path, # Data path
    "testing_log_path" : testing_log_path, # Training log path
    "model_filename" : model_filename, # Model filename
    "streamline_arrays_path" : streamline_arrays_path, # Path to the streamlines array
    
    ####### Misc #######
    "tck_type" : "trk" # TCK type

}

# Define the configuration path and save it as a .json file
configs_folder = "configs_test"
if not os.path.exists(configs_folder):
    os.makedirs(configs_folder, exist_ok=True)
config_path = os.path.join(configs_folder, configs["model_name"] + ".json")

# Save the configuration
dump_json(configs, config_path)

# Load the configuration
configs = load_json(config_path)

run_testing(configs, bias=None)
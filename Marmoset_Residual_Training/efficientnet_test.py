from utils import *
from models import *
from training import *

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
    main_data_path = "/notebooks/model_data_w_resize"
    main_logs_path = "/notebooks/predicted_streamlines"
else:
    main_data_path = "D:\\Brain-MINDS\\model_data"
    main_logs_path = "D:\\Brain-MINDS\\predicted_streamlines"

streamline_arrays_path = os.path.join(main_logs_path, "streamline_predictions", "efficientnet")
training_log_folder = os.path.join(main_logs_path, "training_logs")
model_folder = os.path.join(main_logs_path, "models", "efficientnet")

check_output_folders(streamline_arrays_path, "streamline arrays", wipe=False)
check_output_folders(training_log_folder, "training_log_folder", wipe=False)
check_output_folders(model_folder, "model_folder", wipe=False)

training_log_path = os.path.join(training_log_folder, "efficientnet.csv")
model_filename = os.path.join(model_folder, "efficientnet.h5")

# Create the configs dictionary
configs = {

    ####### Model #######
    "model_name" : "efficientnet", # Model name
    "input_nc" : 1,
    "combination" : True, # Combination
    "task" : "regression_coords", # Task
    "hidden_size" : 32, # number of neurons

    ####### Training #######
    "n_epochs" : 50, # Number of epochs
    "loss" : "MSE_loss", # Loss function
    "optimizer" : "Adam", # Optimizer
    "evaluation_metric" : "MSE_loss", # Evaluation metric
    "shuffle_dataset" : True,
    "separate_hemisphere" : False,
    "cube_size" : 3, # cube size
    "save_best" : True, # Save best model

    ####### Data #######
    "main_data_path" : main_data_path, # Data path
    "training_log_path" : training_log_path, # Training log path
    "model_filename" : model_filename, # Model filename
    "streamline_arrays_path" : streamline_arrays_path, # Path to the streamlines array
    "batch_size" : 8, # Batch size
    "validation_batch_size" : 8, # Validation batch size
    "num_streamlines" : 10, # Number of streamlines to consider from each site
    
    ####### Parameters #######
    "initial_learning_rate" : 0.1, # Initial learning rate
    "early_stopping_patience": None, # Early stopping patience
    "decay_patience": None, # Learning rate decay patience
    "decay_factor": None, # Learning rate decay factor
    "min_learning_rate": 0.1, # Minimum learning rate
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
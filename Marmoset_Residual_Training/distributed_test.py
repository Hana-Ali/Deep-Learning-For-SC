from utils import *
from models import *
from training import *

hpc = False
labs = False
paperspace = False

if hpc:
    main_data_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/model_data"
    main_logs_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/tract_residuals"
elif labs:
    main_data_path = "/media/hsa22/Expansion/Brain_MINDS/model_data"
    main_logs_path = "/media/hsa22/Expansion//Brain_MINDS/tract_residuals"
elif paperspace:
    main_data_path = "/notebooks/model_data_w_resize"
    main_logs_path = "/notebooks/tract_residuals"
else:
    main_data_path = "D:\\Brain-MINDS\\model_data_w_resize"
    main_logs_path = "D:\\Brain-MINDS\\tract_residuals"

residual_arrays_path = os.path.join(main_logs_path, "predicted_residuals")
training_log_path = os.path.join(main_logs_path, "training_logs", "resnet_log.csv")
model_filename = os.path.join(main_logs_path, "models", "resnet_model.pt")

# Create the configs dictionary
configs = {

    ####### Model #######
    "model_name" : "resnet", # Model name
    "input_nc" : 1, # Number of input channels
    "output_nc" : 3, # Number of output channels
    "ngf" : 64, # Number of filters in first conv layer
    "num_blocks" : 3, # Number of residual blocks
    "norm_layer" : "BatchNorm3d", # Normalization layer
    "use_dropout" : False, # Dropout layers
    "padding_type" : "reflect", # Padding type
    
    ####### Training #######
    "n_epochs" : 100, # Number of epochs
    "loss" : "mse_loss", # Loss function
    "optimizer" : "Adam", # Optimizer
    "evaluation_metric" : "MSE_loss", # Evaluation metric
    "shuffle_dataset" : True,
    "separate_hemisphere" : True,
    "save_best" : True, # Save best model
    "regularized" : False, # Regularization
    "vae" : False, # Variational autoencoder

    ####### Data #######
    "main_data_path" : main_data_path, # Data path
    "training_log_path" : training_log_path, # Training log path
    "model_filename" : model_filename, # Model filename
    "residual_arrays_path" : residual_arrays_path, # Path to the residuals array
    "batch_size" : 16, # Batch size
    "validation_batch_size" : 1, # Validation batch size
    
    ####### Parameters #######
    "initial_learning_rate" : 1e-04, # Initial learning rate
    "early_stopping_patience": 50, # Early stopping patience
    "decay_patience": 20, # Learning rate decay patience
    "decay_factor": 0.5, # Learning rate decay factor
    "min_learning_rate": 1e-08, # Minimum learning rate
    "save_last_n_models": 10, # Save last n models

    ####### Misc #######
    "skip_val" : False, # Skip validation

}

# Define the configuration path and save it as a .json file
config_path = os.path.join("configs", configs["model_name"] + ".json")

# Save the configuration
dump_json(configs, config_path)

# Load the configuration
configs = load_json(config_path)

# Define the metric to monitor based on whether we're skipping val or not
if configs["skip_val"]:
    metric_to_monitor = "train_loss_avg"
else:
    metric_to_monitor = "val_loss_avg"

# Define the groups
if configs["skip_val"]:
    groups = ("training",)
else:
    groups = ("training", "validation")

model_metrics = (configs["evaluation_metric"],)

dist_pytorch_training(configs, configs["model_filename"], configs["training_log_path"],
                      configs["residual_arrays_path"], configs["tensorboard_path"],
                      metric_to_monitor=metric_to_monitor,
                      bias=None)
from utils import *
from torch.utils.data import DataLoader
from configs import *
from models import *

if __name__ == '__main__':

        # ------------------------ Load the init configs ------------------------

        # Load the training configs
        config = TrainConfigs().parse()

        # ------------ Data transformation, augmentation and loading ------------

        # Get the mininum pixel value
        min_pixel = int(config.min_pixel * ((config.patch_size[0] * config.patch_size[1] * config.patch_size[2]) / 100))

        # Define the transforms
        train_transforms = [
                Resample(config.new_resolution, config.resample),
                Augmentation(),
                Padding((config.patch_size[0], config.patch_size[1], config.patch_size[2])),
        ]

        # Define the training dataset
        train_dataset = NiftiDataset(data_path=config.train_data_path,
                                        which_direction='AtoB',
                                        transforms=train_transforms,
                                        shuffle_labels=True,
                                        train=True)
        
        # Print the length of the training dataset
        print('Training dataset length: {}'.format(len(train_dataset)))

        # Define the training dataloader
        train_dataloader = DataLoader(train_dataset,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        pin_memory=True)
        
        # ----------------------------- Training -----------------------------

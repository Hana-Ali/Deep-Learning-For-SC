"""
Settings for testing. Imports from base configs
"""

from shared_configs import SharedConfigs

class TestConfigs(SharedConfigs):

    # Initialize the testing configurations
    def initialize(self, parser):

        # Get the shared configurations
        parser = SharedConfigs.initialize(self, parser)

        # Test configurations
        parser.add_argument("--image", type=str, default='./Data_folder/test/images/0.nii')
        parser.add_argument("--result", type=str, default='./Data_folder/test/images/result_0.nii', help='path to the .nii result to save')
        parser.add_argument('--phase', type=str, default='test', help='test')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument("--stride_inplane", type=int, nargs=1, default=32, help="Stride size in 2D plane")
        parser.add_argument("--stride_layer", type=int, nargs=1, default=32, help="Stride size in z direction")

        # Set the defaults
        parser.set_defaults(model='test')

        # Set the train to false
        self.isTrain = False

        # Return the parser
        return parser

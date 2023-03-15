from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # display stuff
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    
        self.parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_steps_freq', type=int, default=1000, help='frequency of saving checkpoints')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--total_iters', type=int, default=100000000, help='# of iter for training')

        # weights
        self.parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')

        self.isTrain = True

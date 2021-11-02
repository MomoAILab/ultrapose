from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--grad_clip_thresh', type=float, default=1.0, help='grad clip thresh')

        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--finetune_model_path', type=str, default=r'', help='finetune_model_path')
        parser.add_argument('--check_per_epoch', type=int, default=5, help='check per iteration')
        parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--total_epoch_num', type=int, default=80, help='total_epoch_num')

        parser.add_argument('--distributed_run', action='store_true', help='distributed_run')
        parser.add_argument('--seed', type=int, default=1234, help='dataset random seed')
        parser.add_argument('--world_size', type=int, default=1, help='num procedure')
        parser.add_argument('--rank', type=int, default=0, help='procedure rank')
        parser.add_argument('--dist_backend', type=str, default=r'nccl', help='dist backend')
        parser.add_argument('--group_name', type=str, default=r'sk_gan', help='dist group name')
        parser.add_argument('--dist_url', type=str, default=r'tcp://localhost:54321', help='dist url')

        parser.add_argument('--use_visdom', action='store_true', help='use visdom')
        parser.add_argument('--visdom_name', type=str, default='TransUltra', help='visdom name')
        parser.add_argument('--visdom_ip', type=str, default='127.0.0.1', help='visdom ip')
        parser.add_argument('--visdom_port', type=int, default=8097, help='visdom port')
        parser.add_argument('--visdom_id', type=str, default='', help='user name')
        parser.add_argument('--visdom_pw', type=str, default='', help='user passwd')

        self.isTrain = True
        return parser

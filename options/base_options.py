import argparse
import dataset

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to train data')
        parser.add_argument('--device', default='cuda', help='device to use for training / testing')
        parser.add_argument('--max_dataset_size', type=int, default=500000, help='max_dataset_size')

        parser.add_argument('--dataset_name', type=str, default='coco', help='[coco]')
        parser.add_argument('--model_name', type=str, default='transformer', help='[transformer]')
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        args, _ = parser.parse_known_args()

        # modify model-related parser options
        # model_name = args.model_name
        # model_option_setter = models.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.isTrain)
        # opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = args.dataset_name
        dataset_option_setter = dataset.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        return self.opt

import importlib
import torch.utils.data
from dataset.base_dataset import BaseDataset
from torch.utils.data.distributed import DistributedSampler
import utils.misc as misc

def find_dataset_using_name(dataset_name):
    dataset_filename = "dataset." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataloader'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(args, is_train):
    dataset_class = find_dataset_using_name(args.dataset_name)
    dataset = dataset_class(args, is_train)
    print("dataset [{}] was created (rank{})".format(type(dataset).__name__, args.rank))

    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
    else:
        if is_train:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler, args.batch_size if is_train else 1, drop_last=True)

    dataloader = torch.utils.data.DataLoader(dataset,
           pin_memory=True,
           batch_sampler=batch_sampler_train,
           num_workers=args.num_threads if is_train else 0)

    return dataloader, dataset


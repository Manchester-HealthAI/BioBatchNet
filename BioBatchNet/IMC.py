import argparse
import collections
import torch
import numpy as np
from torch.utils.data import DataLoader

from parse_config import ConfigParser
import models.model as model
from utils.dataset import GeneralDataset
from utils.util import prepare_device
from utils.trainer import Trainer
import random

def main(config):
    logger = config.get_logger('train')

    # random seed
    SEED = config['seed']
    logger.info(f"random seed is {SEED}")
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    # prepare data
    dataset_name = config['data_loader']['type'] 
    train_dataset = GeneralDataset(dataset_name)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=256)

    # build model
    BioBatchNet = config.init_obj('arch', model)
    logger.info(BioBatchNet)

    # prepare device
    device, device_ids = prepare_device(config['n_gpu'])
    BioBatchNet = BioBatchNet.to(device)
    if len(device_ids) > 1:
        BioBatchNet = torch.nn.DataParallel(BioBatchNet, device_ids=device_ids)

    # train
    optimizer = torch.optim.Adam([
        {'params': BioBatchNet.bio_encoder.parameters(), 'lr': 1e-4},
        {'params': BioBatchNet.batch_encoder.parameters(), 'lr': 1e-4},
        {'params': BioBatchNet.decoder.parameters(), 'lr': 1e-4},
        {'params': BioBatchNet.bio_classifier.parameters(), 'lr': 1e-4},
        {'params': BioBatchNet.batch_classifier.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(config, 
                      model = BioBatchNet, 
                      optimizer = optimizer, 
                      dataloader = train_dataloader,
                      scheduler = lr_scheduler, 
                      device = device)
    
    logger.info("------------------training begin------------------")
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='BatchVAE training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--data', '--data_name'], type=str, target='data_loader;type') 
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

    
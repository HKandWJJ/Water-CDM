import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import time
import model.retinex as rt
from thop import profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/underwater.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')

    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        start = time.time()
        diffusion.test(continous=True)
        end = time.time()
        print('Execution time:', (end - start), 'seconds')
        visuals = diffusion.get_current_visuals(need_LR=False)

        target_img = Metrics.tensor2img(visuals['HR'])
        input_img = Metrics.tensor2img(visuals['INF'])
        restore_img = Metrics.tensor2img(visuals['SR'])

        GMSRBC = False
        if GMSRBC:
            Metrics.save_img(
                #GMSRBC
                rt.adaptive_weighted_fusion(input_img, Metrics.tensor2img(visuals['SR'][-1]), input_img), '{}/{}.png'.format(result_path, idx))
        else :
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}.png'.format(result_path, idx)
            )

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(input_img, Metrics.tensor2img(visuals['SR'][-1]), target_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)

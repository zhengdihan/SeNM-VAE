import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.dataset import Dataset
from .models.model_senmvae import SeNMVAEIR


def main(json_path='./options/train.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    if opt['path']['pretrained_net'] is None:
        init_iter, init_path = option.find_last_checkpoint(opt['path']['models'])
        opt['path']['pretrained_net'] = init_path
        current_step = init_iter
    else:
        current_step = 0
    
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))        
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = SeNMVAEIR(opt)

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    
    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 3) update learning rate
            # -------------------------------
            model.update_learning_rate()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if opt['train']['checkpoint_save'] and current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                                
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['path_L'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    if opt['train']['save_image']:
                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    img_E = util.tensor2uint(visuals['img_E'])
                    img_H = util.tensor2uint(visuals['img_H'])
                    img_L = util.tensor2uint(visuals['img_L'])
                    
                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    if opt['train']['save_image']:
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.imsave(img_E, save_img_path)
                        save_img_H_path = os.path.join(img_dir, '{:s}_H.png'.format(img_name))
                        util.imsave(img_H, save_img_H_path)
                        save_img_L_path = os.path.join(img_dir, '{:s}_L.png'.format(img_name))
                        util.imsave(img_L, save_img_L_path)
                    
                model.save('latest')

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')

if __name__ == '__main__':
    main()

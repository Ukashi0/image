import torch
import numpy as np
import os
import time
from opts.train_opt import TrainOptions
from util.dataLoader import CreateDataLoader
from models import model
from torch.utils.tensorboard import SummaryWriter


def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    config = get_config(opt.config)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(dataset)
    model = model.GanModel()
    model.initialize(opt)
    total_steps = 0
    writer = SummaryWriter('./tensorboard_data')
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()

        print("epoch_start_time:", epoch_start_time)
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            # print("iter_start_time:", iter_start_time)
            total_steps += opt.batchsize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)
            # print("set_input_time:", time.time())
            model.optimize_parameter(epoch)
            # print("optimize time:", time.time())

            # 100次
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_error(epoch)
                print("errors time", time.time())
                # print(errors, "\n")
                t = (time.time() - iter_start_time) / opt.batchsize

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
                # print("save latest time", time.time())

        error = model.get_current_error(epoch)
        for key, value in error.items():
            writer.add_scalar(key, value, epoch)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            print("save latest epoch", time.time())

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if opt.new_lr:
            if epoch == opt.niter:
                model.update_lr()
            elif epoch == (opt.niter + 20):
                model.update_lr()
            elif epoch == (opt.niter + 70):
                model.update_lr()
            elif epoch == (opt.niter + 90):
                model.update_lr()
        else:
            if epoch > opt.niter:
                model.update_lr()

    # writer.close()

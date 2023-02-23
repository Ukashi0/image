import torch
import numpy as np
import os
import time

from opts.test_opt import testOptions
from util.dataLoader import CreateDataLoader
from models import model
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = testOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = model.GanModel()
    model.initialize(opt)
    print("打印模型：", model)

    visualizer = Visualizer(opt)

for i, data in enumerate(dataset):
    model.set_input(data)
    visuals = model.predict()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(visuals, img_path)

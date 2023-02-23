import numpy as np
import os
import ntpath
import time
from . import utils


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.log_name = os.path.join(
            './result/', 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                '================ Training Loss (%s) ================\n' % now)

    # save image to the disk
    def save_images(self, visuals, image_path):
        # image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join('./result/image/', image_name)
            utils.save_image(image_numpy, save_path)

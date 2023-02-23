from .basic import BaseOptions


class testOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--ntest', default=float("inf"), help='# of test examples.')
        # self.parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
        self.parser.add_argument(
            '--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument(
            '--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument(
            '--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--how_many', type=int, default=50, help='how many test images to run')
        self.isTrain = False

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from config import cfg
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel

from model import get_model

exec("from " + cfg.dataset + " import " + cfg.dataset)
print("from base.py > cfg.dataset : ", cfg.dataset)


def get_device():
    """
    Check and return the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        print("Using NVIDIA GPU with CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("from base.py > Using Apple Silicon GPU with MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name="logs.txt"):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name="train_logs.txt")
        self.device = get_device()

    # def get_optimizer(self, model):
    #     if cfg.mode == "rgb+pose":
    #         optimizer = torch.optim.SGD(
    #             list(model.module.aggregator.parameters())
    #             + list(model.module.classifier.parameters()),
    #             lr=cfg.lr,
    #             momentum=cfg.momentum,
    #             weight_decay=cfg.weight_decay,
    #         )
    #     else:
    #         optimizer = torch.optim.SGD(
    #             model.parameters(),
    #             lr=cfg.lr,
    #             momentum=cfg.momentum,
    #             weight_decay=cfg.weight_decay,
    #         )
    #     return optimizer

    def get_optimizer(self, model):
    # Check if the model is wrapped in DataParallel
        if isinstance(model, torch.nn.DataParallel):
            aggregator_params = list(model.module.aggregator.parameters())
            classifier_params = list(model.module.classifier.parameters())
        else:
            aggregator_params = list(model.aggregator.parameters())
            classifier_params = list(model.classifier.parameters())

        if cfg.mode == "rgb+pose":
            optimizer = torch.optim.SGD(
                aggregator_params + classifier_params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )

        return optimizer


    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, "snapshot_{}.pth.tar".format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, "*.pth.tar"))
        cur_epoch = max(
            [
                int(
                    file_name[
                        file_name.find("snapshot_") + 9 : file_name.find(".pth.tar")
                    ]
                )
                for file_name in model_file_list
            ]
        )
        ckpt_path = osp.join(cfg.model_dir, "snapshot_" + str(cur_epoch) + ".pth.tar")
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["network"], strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info("Load checkpoint from {}".format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g["lr"] = cfg.lr / (cfg.lr_dec_factor**idx)
        else:
            for g in self.optimizer.param_groups:
                g["lr"] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g["lr"]
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        # load the dataset class i.e. NTU, KINETICS, MIMETICS
        trainset_loader = eval(cfg.dataset)("train")
        valset_loader = eval(cfg.dataset)("val")

        # train_batch_size = 3
        batch_generator = DataLoader(
            dataset=trainset_loader,
            # batch_size=cfg.num_gpus * cfg.train_batch_size,
            batch_size=1 * cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_thread,
            pin_memory=True,
        )
        val_batch_generator = DataLoader(
            dataset=valset_loader,
            batch_size=1 * cfg.test_batch_size,
            shuffle=True,
            num_workers=cfg.num_thread,
            pin_memory=True,
        )

        # print("train_batch_size : ", cfg.train_batch_size)

        self.trainset = trainset_loader
        
        self.batch_generator = batch_generator
        self.val_batch_generator = val_batch_generator

        self.itr_per_epoch = math.ceil(
            trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size
        )
        self.itr_per_epoch_eval = math.ceil(
            valset_loader.__len__() / cfg.num_gpus / cfg.test_batch_size
        )
        self.class_num = self.trainset.class_num
        self.joint_num = self.trainset.joint_num
        self.skeleton = self.trainset.skeleton

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model(self.class_num, self.joint_num, self.skeleton, "train")
        # model = DataParallel(model).cuda()
        # model = model.to(self.device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)


        # Move the model to the selected device (CUDA, MPS, or CPU)
        # model = model.to(self.device)

        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0

        # print("before model.train()")
        # model.train()
        # print("after model.train()")

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        # print("_make_model() done")


class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name="test_logs.txt")

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.dataset)("test")

        batch_generator = DataLoader(
            dataset=testset_loader,
            batch_size=cfg.num_gpus * cfg.test_batch_size,
            shuffle=False,
            num_workers=cfg.num_thread,
            pin_memory=True,
        )

        self.testset = testset_loader
        self.batch_generator = batch_generator
        self.class_num = self.testset.class_num
        self.joint_num = self.testset.joint_num
        self.skeleton = self.testset.skeleton

    def _make_model(self):
        model_path = os.path.join(
            cfg.model_dir, "snapshot_%d.pth.tar" % self.test_epoch
        )
        assert os.path.exists(model_path), "Cannot find model at " + model_path
        self.logger.info("Load checkpoint from {}".format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(self.class_num, self.joint_num, self.skeleton, "test")
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt["network"], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs):
        self.testset.evaluate(outs)

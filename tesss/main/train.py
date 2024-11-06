import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# print(sys.path)  # Check if /Volumes/SSD/B2C is in the path


import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn
import time
from temp_utilis import logger, bug
from common.logger import colorlogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, dest="gpu_ids")
    parser.add_argument("--mode", type=str, dest="mode")
    parser.add_argument("--continue", dest="continue_train", action="store_true")
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, print("Please set propoer gpu ids")

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    assert args.mode, "please enter mode from one of [rgb_only, pose_only, rgb+pose]."
    return args


def main():
    bug(("START train.py" , ""))
    # argument parse and create log
    # args = parse_args()
    # cfg.set_args("0,1,2,3,4,5", "rgb+pose", False)
    cfg.set_args("0", "rgb+pose", False)
    # cfg.set_args("0", "rgb_only", False)
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        bug(("epoch", epoch))
        bug(("itr_per_epoch", trainer.itr_per_epoch))

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            bug(("ITERATION", itr))
            # bug(("inputs -- dark_video shape", inputs["dark_video"].shape))
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            # trainer.model() is eqv to trainer.model.forward()
            bug("calling trainer.model ...")
            loss = trainer.model(inputs, targets, meta_info, "train")
            print("Called trainer.loss :)")
            bug(("loss['pose_gate'].shape", loss['pose_gate'].shape))
            bug(("loss['action_cls'].shape", loss['action_cls'].shape))
            loss = {k: loss[k].mean() for k in loss}
            {print(f"loss[{k}] : ",loss[k]) for k in loss}
            bug(("loss['pose_gate'].shape", loss['pose_gate'].shape))
            bug(("loss['action_cls'].shape", loss['action_cls'].shape))

            # backward
            sum(loss[k] for k in loss).backward()
            # total_loss = 0 * loss['pose_gate'] + 1* loss['action_cls']
            # total_loss.backward()

            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                "Epoch %d/%d itr %d/%d:"
                % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                "lr: %g" % (trainer.get_lr()),
                "speed: %.2f(%.2fs r%.2f)s/itr"
                % (
                    trainer.tot_timer.average_time,
                    trainer.gpu_timer.average_time,
                    trainer.read_timer.average_time,
                ),
                "%.2fh/epoch"
                % (trainer.tot_timer.average_time / 3600.0 * trainer.itr_per_epoch),
            ]
            screen += ["%s: %.4f" % ("loss_" + k, v.detach()) for k, v in loss.items()]
            trainer.logger.info(" ".join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        trainer.save_model(
            {
                "epoch": epoch,
                "network": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
            },
            epoch,
        )
        torch.cuda.empty_cache()
        time.sleep(0.003)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()

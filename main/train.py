import sys
import os
import logging
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# print(sys.path)  # Check if /Volumes/SSD/B2C is in the path


import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn
import time
from temp_utilis import  bug
from common.logger import colorlogger
from common import metric



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
    # bug(("START train.py" , ""))
    # argument parse and create log
    # args = parse_args()
    # cfg.set_args("0,1,2,3,4,5", "rgb+pose", False)
    cfg.set_args("0", "rgb+pose", False)
    # cfg.set_args("0", "rgb_only", False)
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    # define evaluation metric
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5),)

    # train
    current_best = 0.0
    current_best_top5 = 0.0
    top1_epoch = 0
    top5_epoch = 0

    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.model.train()
        # print("trainer.joint_num : ", trainer.joint_num)
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        total_loss_action_cls = 0
        total_loss_pose_gate = 0
        # bug(("epoch", epoch))
        # bug(("itr_per_epoch", trainer.itr_per_epoch))

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            # bug(("ITERATION", itr))
            # bug(("from train inputs -- dark_video shape", inputs["dark_video"].shape))
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            # trainer.model() is eqv to trainer.model.forward()
            # bug("calling trainer.model ...")
            loss = trainer.model(inputs, targets, meta_info, "train")
            # print("Called trainer.loss .")

            # bug(("loss['pose_gate'].shape", loss['pose_gate'].shape))
            # bug(("loss['action_cls'].shape", loss['action_cls'].shape))

            loss = {k: loss[k].mean() for k in loss}

            # {print(f"loss[{k}] : ",loss[k]) for k in loss}
            # bug(("loss['pose_gate'].shape", loss['pose_gate'].shape))
            # bug(("loss['action_cls'].shape", loss['action_cls'].shape))

            # backward
            sum(loss[k] for k in loss).backward()
            # total_loss = 0 * loss['pose_gate'] + 1* loss['action_cls']
            # total_loss.backward()

            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                "[TRAIN]",
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
            for k,v in loss.items():
                if k == 'action_cls':
                    total_loss_action_cls += v.detach().item()
                elif k == 'pose_gate':
                    total_loss_pose_gate += v.detach().item()


            trainer.logger.info(" ".join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        mean_loss_action_cls = total_loss_action_cls / trainer.itr_per_epoch
        mean_loss_pose_gate = total_loss_pose_gate / trainer.itr_per_epoch
        screen = ["[TRAIN]","Epoch %d/%d :" % (epoch, cfg.end_epoch),"mean_loss_action_cls: %.4f" % (mean_loss_action_cls)]
        screen += ["mean_loss_pose_gate: %.4f" % (mean_loss_pose_gate)]
        trainer.logger.info(" ".join(screen))

        # Save mean losses to CSV
        log_dir  = cfg.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        training_log_file = os.path.join(log_dir, "training_log.csv")
        file_exists = os.path.isfile(training_log_file)

        if not os.path.isfile(training_log_file):
            with open(training_log_file, mode='w') as log_f:
                log_writer = csv.writer(log_f)
                log_writer.writerow(["epoch", "mean_loss_action_cls", "mean_loss_pose_gate"])


        with open(training_log_file, mode='a') as log_f:
            log_writer = csv.writer(log_f)
            log_writer.writerow([epoch, mean_loss_action_cls, mean_loss_pose_gate])

    

        # save model
        trainer.save_model(
            {
                "epoch": epoch,
                "network": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
            },
            epoch,
        )

        # evaluate
        if(trainer.val_batch_generator is not None):
            total_loss_pose_gate = 0
            # print("EVALUATING EPOCH {} ....".format(epoch))
            trainer.logger.info("[VAL] EVALUATING EPOCH {} ....".format(epoch))
            metrics.reset()
            with torch.no_grad():
                trainer.model.eval()
                for itr, (inputs, targets, meta_info) in enumerate(trainer.val_batch_generator):
                    # bug(("from val inputs -- dark_video shape", inputs["dark_video"].shape))
                    
                    trainer.read_timer.toc()
                    trainer.gpu_timer.tic()

                    outs, losses = trainer.model(inputs, targets, meta_info, "test")
                    metrics.update([outs['action_prob'].cpu() ],
                                        targets['action_label'].cpu(),
                                       [losses["action_cls"].cpu() ])

                    losses = {k: losses[k].mean() for k in losses}

                    trainer.optimizer.step()
                    trainer.gpu_timer.toc()
                    
                    for k,v in loss.items():
                        if k == 'action_cls':
                            total_loss_action_cls += v.detach().item()
                        elif k == 'pose_gate':
                            total_loss_pose_gate += v.detach().item()



                    trainer.tot_timer.toc()
                    trainer.tot_timer.tic()
                    trainer.read_timer.tic()
            
            top5_epoch = 0
            top1_epoch = 0
            top1_eval = metrics.get_name_value()[1][0][1]
            top5_eval = metrics.get_name_value()[2][0][1]
            action_cls_loss = metrics.get_name_value()[0][0][1]
            pose_gate_loss = total_loss_pose_gate / trainer.itr_per_epoch_eval

            # Save evaluation metrics to CSV
            eval_log_file = os.path.join(cfg.log_dir, "evaluation_log.csv")
            file_exists = os.path.isfile(eval_log_file)

            if not file_exists:
                with open(eval_log_file, mode='w') as log_f:
                    log_writer = csv.writer(log_f)
                    log_writer.writerow(["epoch", "action_cls_loss", "top1_eval", "top5_eval", "pose_gate_loss"])

            with open(eval_log_file, mode='a') as log_f:
                log_writer = csv.writer(log_f)
                log_writer.writerow([epoch,action_cls_loss, top1_eval, top5_eval, pose_gate_loss])


            screen = [
                        "[VAL]",
                        "Epoch %d/%d "
                        % (epoch, cfg.end_epoch),
                        "lr: %g" % (trainer.get_lr()),
                    ]
            screen += ["%s: %.4f %s: %.4f %s: %.4f %s: %.4f" % ("top1_accuracy", top1_eval, "top5_accuracy", top5_eval, "mean_loss_action_cls", action_cls_loss, "mean_pose_gate", pose_gate_loss)]

            trainer.logger.info(" ".join(screen))

            if (top5_eval > current_best_top5):
                current_best_top5 = top5_eval
                top5_epoch = epoch

            if (top1_eval > current_best) or (top1_eval == current_best):
                current_best = top1_eval
                top1_epoch = epoch


            trainer.logger.info('[VAL] Current best epoch found with top5 accuracy {:.5f} at epoch {:d}, saved'.format(current_best_top5, top5_epoch))
            trainer.logger.info('[VAL] Current best epoch found with top1 accuracy {:.5f} at epoch {:d}, saved'.format(current_best, top1_epoch))


            torch.set_grad_enabled(True) # for pytorch040 version



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

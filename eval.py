import os
import yaml
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.custom_dataset_data_loader import CustomDatasetDataLoader, sample_data
from options.eval_options import parser
from utils.saving_utils import load_checkpoint_mgpu
from utils.miou import intersect_and_union
from networks import U2NET
from tqdm import tqdm


def options_printing_saving(opt):
    os.makedirs(opt.logs_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "checkpoints"), exist_ok=True)

    # Saving options in yml file
    option_dict = vars(opt)
    with open(os.path.join(opt.save_dir, "training_options.yml"), "w") as outfile:
        yaml.dump(option_dict, outfile)

    for key, value in option_dict.items():
        print(key, value)


def eval_loop(opt):
    device = torch.device("cuda:0")

    u_net = U2NET(in_ch=3, out_ch=4)
    u_net = load_checkpoint_mgpu(u_net, opt.unet_checkpoint)
    u_net = u_net.to(device)
    u_net.eval()

    with open(os.path.join(opt.save_dir, "networks.txt"), "w") as outfile:
        print("<----U-2-Net---->", file=outfile)
        print(u_net, file=outfile)

    custom_dataloader = CustomDatasetDataLoader()
    opt.serial_batches=True
    custom_dataloader.initialize(opt)
    loader = custom_dataloader.get_loader()

    dataset_size = len(custom_dataloader)
    print("Total number of images: %d" % dataset_size)

    # loss function
    weights = np.array([1, 1.5, 1.5, 1.5], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)

    get_data = sample_data(loader)

    # Main training loop
    ious, losses = [], []

    for _ in tqdm(range(dataset_size//opt.batchSize)):
        data_batch = next(get_data)
        image, label = data_batch
        image = Variable(image.to(device))
        label = label.type(torch.long)
        label = Variable(label.to(device))

        d0, d1, d2, d3, d4, d5, d6 = u_net(image)

        loss0 = loss_CE(d0, label)
        loss1 = loss_CE(d1, label)
        loss2 = loss_CE(d2, label)
        loss3 = loss_CE(d3, label)
        loss4 = loss_CE(d4, label)
        loss5 = loss_CE(d5, label)
        loss6 = loss_CE(d6, label)
        del d1, d2, d3, d4, d5, d6

        total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        d0 = F.log_softmax(d0, dim=1)
        # d0 = torch.max(d0, dim=1, keepdim=True)[1]
        seg_pred = d0.argmax(dim=1).squeeze(1)

        area_intersect, area_union, _, _ = intersect_and_union(
            pred_label=seg_pred.cpu().numpy(),
            label=label.cpu().numpy(),
            num_classes=3,
            ignore_index=255,
            label_map=None,
            reduce_zero_label=True,
        )
        iou = area_intersect / area_union

        ious.append(iou)
        losses.append(total_loss.cpu().detach())

    class_miou = torch.stack(ious).nanmean(dim=0)
    tot_miou = class_miou.nanmean()
    mean_loss = torch.stack(losses).nanmean()

    print("Classwise mIoU:", class_miou)
    print("mIoU:", tot_miou)
    print("Average Loss:", mean_loss)


if __name__ == "__main__":
    opt = parser()
    options_printing_saving(opt)
    eval_loop(opt)
    print("All done!")

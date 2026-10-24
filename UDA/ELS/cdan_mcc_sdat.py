import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
import re
# import wandb
import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(1, str(CURRENT_DIR.parent))
from imagefolder_splits import load_imagefolder, remap_class_indices
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, a_distance
from common.utils.sam import SAM

import utils


class AlignedImageFolder(Dataset):
    """Wrap an ImageFolder and remap class indices to a reference class_to_idx."""

    def __init__(self, base_dataset: ImageFolder, ref_class_to_idx: dict):
        self.base = base_dataset
        self.ref_class_to_idx = ref_class_to_idx
        self.transform = base_dataset.transform
        self.target_transform = base_dataset.target_transform
        self.loader = base_dataset.loader

        remap = remap_class_indices(base_dataset.class_to_idx, ref_class_to_idx)

        self.samples = []
        dropped = 0
        for path, old_idx in base_dataset.samples:
            if old_idx in remap:
                self.samples.append((path, remap[old_idx]))
            else:
                dropped += 1

        self.targets = [t for _, t in self.samples]
        self.classes = [None] * len(ref_class_to_idx)
        for cls_name, idx in ref_class_to_idx.items():
            self.classes[idx] = cls_name
        self.class_to_idx = dict(ref_class_to_idx)

        if dropped > 0:
            warnings.warn(f"Dropped {dropped} samples with classes not present in source mapping.")
        if len(self.samples) == 0:
            raise ValueError(
                "After class remapping, no samples remain in target dataset. "
                "Check class folder naming consistency between source and target directories."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def _resolve_imagefolder_root(path_str: str) -> str:
    """Accept either a class-folder root or a parent domain root containing `images/`."""
    path = Path(path_str)
    images_child = path / "images"
    if images_child.is_dir():
        return str(images_child)

    if path.is_dir():
        return str(path)

    if path.name.lower() == "images" and path.parent.is_dir():
        return str(path.parent)

    raise FileNotFoundError(
        f"ImageFolder root not found: {path}. Expected either that directory itself "
        f"or an 'images' subdirectory under it."
    )


def get_custom_path_datasets(args: argparse.Namespace, train_transform, val_transform):
    source_train_dirs = []
    if args.source_train_dir:
        source_train_dirs.append(args.source_train_dir)
    if args.source_train_dirs:
        source_train_dirs.extend(args.source_train_dirs)

    # De-duplicate while preserving order so accidental repeats do not inflate samples.
    source_train_dirs = list(dict.fromkeys(source_train_dirs))

    if not source_train_dirs or not args.target_test_dir:
        raise ValueError(
            "Custom path mode requires at least one source dir via --source-train-dir/--source-train-dirs "
            "and --target-test-dir."
        )

    target_train_dir = args.target_train_dir if args.target_train_dir else args.target_test_dir

    resolved_source_train_dirs = [_resolve_imagefolder_root(p) for p in source_train_dirs]
    resolved_target_train_dir = _resolve_imagefolder_root(target_train_dir)
    resolved_target_test_dir = _resolve_imagefolder_root(args.target_test_dir)

    raw_source_datasets = [ImageFolder(src_dir, transform=train_transform) for src_dir in resolved_source_train_dirs]
    source_ref_dataset = raw_source_datasets[0]
    source_class_to_idx = source_ref_dataset.class_to_idx
    aligned_source_datasets = [AlignedImageFolder(ds, source_class_to_idx) for ds in raw_source_datasets]

    if len(aligned_source_datasets) == 1:
        train_source_dataset = aligned_source_datasets[0]
    else:
        train_source_dataset = ConcatDataset(aligned_source_datasets)

    raw_train_target_dataset = load_imagefolder(
        resolved_target_train_dir, transform=train_transform, split_file=args.target_train_list
    )
    raw_val_dataset = load_imagefolder(
        resolved_target_test_dir, transform=val_transform, split_file=args.target_test_list
    )
    raw_test_dataset = load_imagefolder(
        resolved_target_test_dir, transform=val_transform, split_file=args.target_test_list
    )

    class_names = source_ref_dataset.classes
    num_classes = len(class_names)

    train_target_dataset = AlignedImageFolder(raw_train_target_dataset, source_class_to_idx)
    val_dataset = AlignedImageFolder(raw_val_dataset, source_class_to_idx)
    test_dataset = AlignedImageFolder(raw_test_dataset, source_class_to_idx)

    if raw_train_target_dataset.classes != class_names:
        warnings.warn(
            "Class order mismatch detected for target-train; labels are remapped to source class order."
        )
    if raw_test_dataset.classes != class_names:
        warnings.warn(
            "Class order mismatch detected for target-test; labels are remapped to source class order."
        )

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def main(args: argparse.Namespace, eps=0.):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    # if args.log_results:
    #     wandb.init(project="ELS", name=args.log_name)
    #     wandb.config.update(args)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = args.seed is None
    device = args.device

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    if args.source_train_dir or args.source_train_dirs or args.target_test_dir:
        print("[INFO] Using custom path datasets (ImageFolder). --data/--source/--target are ignored.")
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
            get_custom_path_datasets(args, train_transform, val_transform)
    elif args.dataset_ratio == -1:
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
            utils.get_dataset(args.data, args.root, args.source,
                              args.target, train_transform, val_transform)
    else:
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
            utils.get_ratio_dataset(args.data, args.root, args.source,
                                    args.target, train_transform, val_transform, args.dataset_ratio)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.phase in {'train', 'finetune'} and (len(train_source_loader) == 0 or len(train_target_loader) == 0):
        raise ValueError("Source and target training sets must each contain at least --batch-size images.")

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch and args.phase in {'train', 'finetune'})
    # print(backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    # print(pool_layer)
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    if args.randomized:
        domain_discri = DomainDiscriminator(
            args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(
            classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    base_optimizer = torch.optim.SGD
    ad_optimizer = SGD(domain_discri.get_parameters(
    ), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer = SAM(classifier.get_parameters(), base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr *
                                                 (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_ad = LambdaLR(
        ad_optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim, eps=eps
    ).to(device)

    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path(f'best_with_eps{eps}'), map_location='cpu')
        # checkpoint = torch.load(path, map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # resume from the given checkpoint
    if args.phase == 'finetune':
        path = args.weight_path
        print(f"[INFORMATION] Using the weights stored at {args.weight_path}")
        classifier.load_state_dict(torch.load(path))

    # analysis the model
    if args.phase == 'analysis':
        path = args.weight_path
        print(f"[INFORMATION] Using the weights stored at {args.weight_path}")
        classifier.load_state_dict(torch.load(path))
        # extract features from both domains
        feature_extractor = nn.Sequential(
            classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(
            train_source_loader, feature_extractor, device)
        print(source_feature.size())
        target_feature = collect_feature(
            train_target_loader, feature_extractor, device)
        A_distance = a_distance.calculate(
            source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return 1


    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return acc1

    all_accuracy = []

    # start training
    best_acc1 = float('-inf')
    for epoch in range(args.epochs):
        print("lr_bbone:", lr_scheduler.get_last_lr()[0])
        print("lr_btlnck:", lr_scheduler.get_last_lr()[1])

        [acc, loss] = train(train_source_iter, train_target_iter, classifier, domain_adv, mcc_loss, optimizer,
                            ad_optimizer,
                            lr_scheduler, lr_scheduler_ad, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)
        all_accuracy.append([acc1, loss])

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path(f'best_with_eps{eps}'))
        best_acc1 = max(acc1, best_acc1)
    all_accuracy = np.array(all_accuracy)
    np.save(os.path.join(args.log, "acc_and_loss"), all_accuracy)
    print("best_acc1 = {:3.2f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path(f'best_with_eps{eps}')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.2f}".format(acc1))
    return acc1


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_adv: ConditionalDomainAdversarialLoss, mcc, optimizer, ad_optimizer,
          lr_scheduler: LambdaLR, lr_scheduler_ad, epoch: int, args: argparse.Namespace):
    device = args.device
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        ad_optimizer.zero_grad()

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        mcc_loss_value = mcc(y_t)
        loss = cls_loss + mcc_loss_value

        loss.backward()

        # Calculate  (w) and add it to the weights
        optimizer.first_step(zero_grad=True)

        # Calculate task loss and domain loss
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(y_s, f_s, y_t, f_t) + mcc(y_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]
        # if args.log_results:
        #     wandb.log({'iteration': epoch * args.iters_per_epoch + i, 'loss': loss, 'cls_loss': cls_loss,
        #                'transfer_loss': transfer_loss, 'domain_acc': domain_acc})

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        domain_accs.update(domain_acc, x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        loss.backward()
        # Update parameters of domain classifier
        ad_optimizer.step()
        # Update parameters (Sharpness-Aware update)
        optimizer.second_step(zero_grad=True)
        lr_scheduler.step()
        lr_scheduler_ad.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    # epoch
    return [cls_accs.avg.cpu().numpy(), losses.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDAN+MCC with SDAT for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--save-acc', action='store_true',
                        help='whether to save acc')
    parser.add_argument('root', nargs='?', default='', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--eps', default=1.0, type=float,
                        help='hyper-parameter for environemnt label smoothing.')
    parser.add_argument('--dataset-ratio', default=-1, type=float,
                        help='The ratio between dataset1 and dataset2, it will crop the last dataset in the given '
                             'datasets rate')
    parser.add_argument('--source-train-dir', type=str, default='',
                        help='Custom source train directory. Must be an ImageFolder-style path.')
    parser.add_argument('--source-train-dirs', type=str, nargs='+', default=[],
                        help='Custom source train directories to be concatenated for training.')
    parser.add_argument('--target-train-dir', type=str, default='',
                        help='Custom target train directory. If empty, --target-test-dir is used.')
    parser.add_argument('--target-test-dir', type=str, default='',
                        help='Custom target test directory. Must be an ImageFolder-style path.')
    parser.add_argument('--target-train-list', type=str, default=None,
                        help='Official target train split (image_path label), resolved against --target-train-dir.')
    parser.add_argument('--target-test-list', type=str, default=None,
                        help='Official target test split (image_path label), resolved against --target-test-dir.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False,
                        action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,#1000
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--weight_path", type=str, default='cdan',
                        help="Path to the saved weights")
    parser.add_argument("--phase", type=str, default='train',
                        choices=['train', 'test', 'analysis', 'finetune', 'analysis_distance'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--log_results', action='store_true',
                        help="To log results in wandb")
    parser.add_argument('--gpu', type=str, default="1", help="GPU ID")
    parser.add_argument('--log_name', type=str,
                        default="log", help="log name for wandb")
    parser.add_argument('--rho', type=float, default=0.05, help="GPU ID")
    parser.add_argument('--temperature', default=2.0,
                        type=float, help='parameter temperature scaling')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.eps >= 1:
        epses = [0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]
        acces = np.array([0. for i in epses])
        for i in range(len(epses)):
            print('eps as these epoch is', epses[i])
            acces[i] = main(args, eps=epses[i])
        idx = acces.argmax()
        print(f'the best result is {acces[idx]} with eps {epses[idx]}')
        print(f'when eps=0, the result is', acces[0])
    else:
        print('eps as these epoch is', args.eps)
        main(args, eps=args.eps)
        exit()

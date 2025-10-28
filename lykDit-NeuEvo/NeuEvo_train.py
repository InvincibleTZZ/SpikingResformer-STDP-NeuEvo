import os
import sys
import time
import logging
import torch
import utils as dutils
import argparse
import numpy as np
import torch.utils
import torch.nn as nn
import genotypes
from NeuEvo_model import NetworkCIFAR as Network
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from thop import profile
import matplotlib.pyplot as plt
import json
from datasets import *
from braincog.base.utils.criterions import UnilateralMse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data/datasets',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset|cifar10|cifar100|mnist|fashion-mnist')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--device', type=int, default=0, help='gpu device id')
parser.add_argument('--multi-gpus', action='store_true',
                    default=False, help='use multi gpus')
parser.add_argument('--parse_method', type=str,
                    default='darts', help='experiment name')
parser.add_argument('--epochs', type=int, default=150,
                    help='num of training epochs')
parser.add_argument('--init-channels', type=int,
                    default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true',
                    default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float,
                    default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true',
                    default=False, help='use auto augmentation')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--arch', type=str, default='dvsc10_new0',
                    help='which architecture to use')           #######genotypes
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--img_size', default=32, type=int)
parser.add_argument('--step', default=4, type=int,
                    help='time steps for SNN (recommended: 4-8, default: 4)')
parser.add_argument('--node-type', default='LIFNode', type=str)
parser.add_argument('--suffix', default='', type=str)

parser.add_argument('--encode', type=str, default='direct', 
                    help='input encoding type (direct/ttfs/rate)')

parser.add_argument('--optimizer', type=str, default='adamw', 
                    choices=['sgd', 'adamw', 'adam'],
                    help='optimizer type (sgd/adamw/adam)')
parser.add_argument('--loss-fn', type=str, default='ce', 
                    choices=['ce', 'mse'],
                    help='loss function: ce=CrossEntropy, mse=UnilateralMse')
parser.add_argument('--auto-lr', action='store_true',
                    default=False, help='auto adjust learning rate based on dataset and optimizer')


class TrainNetwork(object):
    """The main train network"""

    def __init__(self, args):
        super(TrainNetwork, self).__init__()
        self.args = args
        self.dur_time = 0

        self.train_acc_history = []
        self.valid_acc_history = []
        self.train_loss_history = []
        self.valid_loss_history = []
        self.epochs_list = []

        self._init_log()
        self._init_device()
        self._init_data_queue()
        self._init_model()

    def _init_log(self):
        self.args.save = '/data/floyed/darts/logs/eval/' + self.args.arch + '/' + 'cifar10' + '/eval-{}-{}-{}'.format(
            self.args.save, time.strftime('%Y%m%d-%H%M'), self.args.suffix)
        dutils.create_exp_dir(self.args.save, scripts_to_save=None)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger('Architecture Training')
        self.logger.addHandler(fh)

    def _init_device(self):
        if not torch.cuda.is_available():
            self.logger.info('no gpu device available')
            sys.exit(1)
        np.random.seed(self.args.seed)
        self.device_id = self.args.device
        self.device = torch.device('cuda:{}'.format(
            0 if self.args.multi_gpus else self.device_id))
        cudnn.benchmark = True
        torch.manual_seed(self.args.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(self.args.seed)
        logging.info('gpu device = %d' % self.args.device)
        logging.info("args = %s", self.args)

    def _init_data_queue(self):
        if self.args.dataset in ['mnist', 'fashion-mnist']:
            img_size = 32
            import torchvision.transforms as transforms
            train_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的归一化参数
            ])
            valid_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            img_size = args.img_size
            train_transform = build_transform(True, args.img_size)
            valid_transform = build_transform(False, args.img_size)
          
        if self.args.dataset == 'cifar10':
            train_data = dset.CIFAR10(
                root=self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR10(
                root=self.args.data, train=False, download=True, transform=valid_transform)
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            train_data = dset.CIFAR100(
                root=self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR100(
                root=self.args.data, train=False, download=True, transform=valid_transform)
            self.num_classes = 100
        elif self.args.dataset == 'mnist':
            train_data = dset.MNIST(
                root=self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.MNIST(
                root=self.args.data, train=False, download=True, transform=valid_transform)
            self.num_classes = 10
        elif self.args.dataset == 'fashion-mnist':
            train_data = dset.FashionMNIST(
                root=self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.FashionMNIST(
                root=self.args.data, train=False, download=True, transform=valid_transform)
            self.num_classes = 10
        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        self.valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    def _init_model(self):
        genotype = eval('genotypes.%s' % self.args.arch)
        model = Network(self.args.init_channels,
                        self.num_classes,
                        self.args.layers,
                        self.args.auxiliary,
                        genotype,
                        dataset=args.dataset,
                        
                        encode_type=args.encode,
                        step=args.step,
                        node_type=args.node_type
                       
                        
                        )
        if self.args.dataset in ['mnist', 'fashion-mnist']:
            # MNIST/Fashion-MNIST: 1通道，28x28尺寸
            sample_input = torch.randn(1, 1, 32, 32)
        elif self.args.dataset in ['dvsg', 'dvsc10', 'NCALTECH101']:
            # DVS数据集: 2通道，具体尺寸可能不同
            sample_input = torch.randn(1, 2, 32, 32)
        else:
            # CIFAR-10/CIFAR-100等: 3通道，32x32尺寸
            sample_input = torch.randn(1, 3, 32, 32)
        
        flops, params = profile(model, inputs=(sample_input,), verbose=False)
        self.logger.info('flops = %fM', flops / 1e6)
        self.logger.info('param size = %fM', params / 1e6)

        # Try move model to multi gpus
        if torch.cuda.device_count() > 1 and self.args.multi_gpus:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)
        self.model = model.to(self.device)

        # Auto adjust learning rate if enabled
        if self.args.auto_lr:
            if self.args.optimizer.lower() == 'sgd':
                # SGD can use larger learning rates
                if self.args.dataset in ['mnist', 'fashion-mnist']:
                    self.args.learning_rate = 0.01  # Smaller for MNIST
                else:
                    self.args.learning_rate = 0.025  # Standard for CIFAR
            else:  # AdamW or Adam
                # Adaptive optimizers need smaller learning rates
                if self.args.dataset in ['mnist', 'fashion-mnist']:
                    self.args.learning_rate = 0.0005  # Smaller for MNIST
                else:
                    self.args.learning_rate = 0.001  # Standard for CIFAR
            self.logger.info('Auto-adjusted learning rate to %e', self.args.learning_rate)

        # Select loss function
        if self.args.loss_fn == 'mse':
            criterion = UnilateralMse(1.)
            self.logger.info('Using UnilateralMse loss function')
        else:
            criterion = nn.CrossEntropyLoss()
            self.logger.info('Using CrossEntropyLoss loss function')
        self.criterion = criterion.to(self.device)

        # Select optimizer
        if self.args.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
            self.logger.info('Using SGD optimizer')
        elif self.args.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
            self.logger.info('Using Adam optimizer')
        else:  # adamw
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
            self.logger.info('Using AdamW optimizer')

        self.best_acc_top1 = 0
        # optionally resume from a checkpoint
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint {}".format(self.args.resume))
                checkpoint = torch.load(
                    self.args.resume, map_location=self.device)
                self.dur_time = checkpoint['dur_time']
                self.args.start_epoch = checkpoint['epoch']
                self.best_acc_top1 = checkpoint['best_acc_top1']
                self.args.drop_path_prob = checkpoint['drop_path_prob']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(self.args.epochs), eta_min=0,
                                                                    last_epoch=-1 if self.args.start_epoch == 0 else self.args.start_epoch)
        # reload the scheduler if possible
        if self.args.resume and os.path.isfile(self.args.resume):
            checkpoint = torch.load(self.args.resume)
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def run(self):
        self.logger.info('args = %s', self.args)
        run_start = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):

            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.scheduler.get_lr()[0]
            self.logger.info('epoch % d / %d  lr %e', epoch,
                             self.args.epochs, current_lr)

            self.model.drop_path_prob = self.args.drop_path_prob * epoch / self.args.epochs

            train_acc, train_obj = self.train()
            self.logger.info('train loss %e, train acc %f',
                             train_obj, train_acc)

            valid_acc_top1, valid_acc_top5, valid_obj = self.infer()
            self.logger.info('valid loss %e, top1 valid acc %f top5 valid acc %f',
                             valid_obj, valid_acc_top1, valid_acc_top5)
            self.logger.info('best valid acc %f', self.best_acc_top1)

            self.epochs_list.append(epoch)
            self.train_acc_history.append(train_acc)
            self.valid_acc_history.append(valid_acc_top1)
            self.train_loss_history.append(train_obj)
            self.valid_loss_history.append(valid_obj)

            self.scheduler.step()

            is_best = False
            if valid_acc_top1 > self.best_acc_top1:
                self.best_acc_top1 = valid_acc_top1
                is_best = True

            dutils.save_checkpoint({
                'epoch': epoch + 1,
                'dur_time': self.dur_time + time.time() - run_start,
                'state_dict': self.model.state_dict(),
                'drop_path_prob': self.args.drop_path_prob,
                'best_acc_top1': self.best_acc_top1,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }, is_best, self.args.save)

            torch.cuda.empty_cache()

        self.logger.info('train epoches %d, best_acc_top1 %f, dur_time %s',
                         self.args.epochs, self.best_acc_top1,
                         dutils.calc_time(self.dur_time + time.time() - run_start))

    def train(self):
        objs = dutils.AvgrageMeter()
        top1 = dutils.AvgrageMeter()
        top5 = dutils.AvgrageMeter()

        self.model.train()

        for step, (input, target) in enumerate(self.train_queue):

            

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            self.optimizer.zero_grad()
            logits, logits_aux = self.model(input)
            loss = self.criterion(logits, target)
            if self.args.auxiliary:
                loss_aux = self.criterion(logits_aux, target)
                loss += self.args.auxiliary_weight * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            if step % 50 == 0:
                torch.cuda.empty_cache()

            prec1, prec5 = dutils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0:
                self.logger.info('train %03d %e %f %f', step,
                                 objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    def infer(self):
        objs = dutils.AvgrageMeter()
        top1 = dutils.AvgrageMeter()
        top5 = dutils.AvgrageMeter()
        self.model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(self.valid_queue):
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                logits, _ = self.model(input)
                loss = self.criterion(logits, target)

                prec1, prec5 = dutils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % self.args.report_freq == 0:
                    self.logger.info('valid %03d %e %f %f',
                                     step, objs.avg, top1.avg, top5.avg)
            return top1.avg, top5.avg, objs.avg
        
    def _plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.epochs_list) == 0:
            return
            
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 准确率曲线
        ax1.plot(self.epochs_list, self.train_acc_history, 'b-', label='训练准确率', linewidth=2)
        ax1.plot(self.epochs_list, self.valid_acc_history, 'r-', label='验证准确率', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_title('训练和验证准确率曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 2. 损失曲线
        ax2.plot(self.epochs_list, self.train_loss_history, 'b-', label='训练损失', linewidth=2)
        ax2.plot(self.epochs_list, self.valid_loss_history, 'r-', label='验证损失', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('损失值')
        ax2.set_title('训练和验证损失曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 训练准确率详细图
        ax3.plot(self.epochs_list, self.train_acc_history, 'g-', marker='o', markersize=4, label='训练准确率')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('训练准确率 (%)')
        ax3.set_title('训练准确率详细曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. 验证准确率详细图
        ax4.plot(self.epochs_list, self.valid_acc_history, 'orange', marker='s', markersize=4, label='验证准确率')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('验证准确率 (%)')
        ax4.set_title('验证准确率详细曲线')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.args.save, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f'训练曲线已保存到: {plot_path}')
        
        # 打印当前最佳结果
        if len(self.valid_acc_history) > 0:
            best_epoch = np.argmax(self.valid_acc_history)
            best_acc = max(self.valid_acc_history)
            self.logger.info(f'当前最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch + 1})')

    def _save_training_data(self):
        """保存训练数据到JSON文件"""
        training_data = {
            'epochs': self.epochs_list,
            'train_accuracy': self.train_acc_history,
            'valid_accuracy': self.valid_acc_history,
            'train_loss': self.train_loss_history,
            'valid_loss': self.valid_loss_history,
            'best_valid_acc': self.best_acc_top1,
            'config': {
                'arch': self.args.arch,
                'dataset': self.args.dataset,
                'init_channels': self.args.init_channels,
                'layers': self.args.layers,
                'step': self.args.step,
                'node_type': self.args.node_type,
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'epochs': self.args.epochs
            }
        }
        
        # 保存到JSON文件
        json_path = os.path.join(self.args.save, 'training_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f'训练数据已保存到: {json_path}')


if __name__ == '__main__':
    args = parser.parse_args()
    train_network = TrainNetwork(args)
    train_network.run()
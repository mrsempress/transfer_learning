# encoding=utf-8
"""
    Created on 13:42 2019/07/29
    @author: Chenxi Huang
    Integrate code into a unified framework using argparse.
"""
import argparse
from JDA import JDA, JDA_2
from ResNet50 import ResNet
from Mnist2Usps import Baseline
from AssociativeDA import train as ADA
from D_Adversarial_NN import DANN
from MCD_UDA import main as MCD
from MADA import MADA
from Self_ensemble import self_ensemble
from TCA import TCA

parser = argparse.ArgumentParser(description='Transfer learning by hcx')

parser.add_argument('--model', type=str, default='TCA')
parser.add_argument('--network', type=str, default='resnet')
parser.add_argument('--optimizer', type=str, default='SGD', help='which optimizer')
parser.add_argument('--gpu', type=str, default='3', help='Specify GPU')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--test_interval', type=int, default=500)

parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--n_classes', type=int, default=31)
parser.add_argument('--epochs', type=int, default=256, help="Number of Epochs")
parser.add_argument('--iterations', type=int, default=256)
parser.add_argument('--gamma', type=float, default=10)
parser.add_argument('--kernel', type=str, default='primal')
parser.add_argument('--lambd', type=float, default=1.0)
parser.add_argument('--learningrate', default=3e-4, type=float, help="Learning rate for Adam. Defaults to "
                                                                     "Karpathy's constant ;-)")

# Domain Adaptation Args
parser.add_argument('--dataset', type=str, default='Office31')
parser.add_argument('--source', type=str, default='amazon', help="Source Dataset")
parser.add_argument('--target', type=str, default='webcam', help="Target Dataset")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--sourcebatch', type=int, default=100, help="Batch size of Source")
parser.add_argument('--targetbatch', type=int, default=1000, help="Batch size of Target")

# JDA Hyperparams
parser.add_argument('--k', type=int, default=100)

#  Self_ensemble Hyperparams
parser.add_argument('--rgb', type=str, default='False')
parser.add_argument('--teacher_alpha', type=float, default=0.99)
parser.add_argument('--fix_ema', type=str, default='False')
parser.add_argument('--cls_balance_loss', type=str, default='bce')
parser.add_argument('--combine_batches', type=str, default='False')
parser.add_argument('--epochs_size', type=str, default='target')

# DANN Hyperparams
parser.add_argument('--use_msda', type=str, default='True')
parser.add_argument('--use_adversarial', type=str, default='True')

# Associative DA Hyperparams
parser.add_argument('--checkpoint', default="", help="Checkpoint path")
parser.add_argument('--visit', type=float, default=0.1, help="Visit weight")
parser.add_argument('--walker', type=float, default=1.0, help="Walker weight")

# MCD Hyperparams
# parser.add_argument('--all_use', type=str, default='no', metavar='N',
#                     help='use all training data? in usps adaptation')
parser.add_argument('--all_use', type=str, default='no',
                    help='use all training data? in usps adaptation')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', type=str, default='False',
                    help='evaluation only option')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--one_step', type=str, default='False',
                    help='one step training with gradient reversal layer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', type=str, default='False',
                    help='save_model or not')
parser.add_argument('--use_abs_diff', type=str, default='False',
                    help='use absolute difference value as a measurement')

# MADA Hyperparams
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.9)

args = parser.parse_args()


if __name__ == '__main__':
    print(args.model)
    if args.model == 'TCA':
        # python main.py --model='TCA' --source='amazon' --target='webcam'
        TCA.work(args.source, args.target, args.gpu)

    elif args.model == 'JDA1':
        """
        srcStr = ['Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam',
                'dslr', 'dslr', 'dslr']
        tgtStr = ['amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10',
              'amazon', 'webcam']
        """
        # python main.py --model='JDA1' --source='amazon' --target='webcam' --gamma=1.0
        JDA.work(args.source, args.target, args.gpu, args.k, args.lambd, args.kernel, args.gamma)

    elif args.model == 'JDA2':
        # python main.py --model='JDA2' --source='amazon' --target='webcam' --gamma=1.0
        JDA_2.work(args.source, args.target, args.gpu, args.lambd, args.kernel, args.gamma)

    elif args.model == 'Baseline_Digit':
        # python main.py --model='Baseline_Digit' --source='MNIST' --target='USPS' --n_classes=10 /
        # --batch_size=256 --num_workers=4 --learningrate=0.001 --epochs=25
        Baseline.work(args.source, args.target, args.gpu, args.n_classes, args.batch_size,
                      args.num_workers, args.learningrate, args.epochs)

    elif args.model == 'Baseline_Office_31':
        # python main.py --model='Baseline_Office_31' --source='webcam' --target='dslr' --network='resnet'
        ResNet.work(args.source, args.target, args.network, args.gpu, args.seed, args.batch_size)

    elif args.model == 'Self_ensemble':
        # python main.py --model='Self_ensemble' --source='mnist' --target='usps' --rgb='False' --seed=0
        self_ensemble.work(args.source, args.target, args.rgb, args.gpu, args.teacher_alpha, args.fix_ema,
                           args.cls_balance_loss, args.combine_batches, args.epochs, args.batch_size, args.seed,
                           args.learningrate, args.epochs_size)

    elif args.model == 'DANN':
        # use msda or not and use adversarial or not
        # python main.py --model='DANN' --use_msda='True' --use_adversarial='True'
        DANN.work(args.use_msda, args.use_adversarial)

    elif args.model == 'ADA':
        # python main.py --model='ADA' --source='mnist' --target='svhn' --epochs=1000 \
        # --gpu='3' --num_workers=4
        # the source and target must be mnist and svhn.
        ADA.work(args.source, args.target, args.gpu, args.sourcebatch, args.targetbatch, args.checkpoint, args.visit,
                 args.walker, args.epochs, args.learningrate, args.num_workers)

    elif args.model == 'MCD':
        # python main.py --model='MCD' --source='mnist' --target='svhn' --batch_size=128 --learningrate=0.0002 \
        # --seed=1 --optimizer='adam' --epochs=200
        MCD.work(args.source, args.target, args.gpu, args.epochs, args.resume_epoch, args.learningrate, args.batch_size,
                 args.optimizer, args.num_k, args.all_use, args.checkpoint_dir, args.save_epoch, args.save_model,
                 args.one_step, args.use_abs_diff, args.eval_only)

    elif args.model == 'MADA':
        # python main.py --model='MADA' --source='amazon' --target='dslr' --gpu=2
        # --num_workers=4
        MADA.work(args.source, args.target, args.gpu, args.batch_size, args.epochs, args.learningrate, args.gamma,
                  args.optimizer, args.test_interval, args.iterations, args.weight_decay,
                  args.momentum, args.num_workers)

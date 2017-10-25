'''
 This file manages the training process and saves
 the training results and metrics.

'''

import argparse

# Set the frequently changing parameters from outside.
parser = argparse.ArgumentParser(description="Marble GAN training algorithm")

parser.add_argument("--iteration", type=int, default=5, metavar='N',
                    help='the number of overall iterations (including sample generation)')
parser.add_argument("--lr", type=float, default=0.0001, metavar='N',
                    help="learning rate")
parser.add_argument("--batch-size", type=int, default=8, metavar='N',
                    help="traditional batch size")
parser.add_argument("--epochs", type=int, default=1, metavar='N',
                    help="traditional epoch")
parser.add_argument("--eval-file-name", default="eval.csv", metavar='S',
                    help="the name of the file ")

args = parser.parse_args()




import pandas as pd
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Showing results.")
parser.add_argument("--file-name",  default="eval.csv", metavar='S')
parser.add_argument("--mode", default=1, type=int, metavar='N')

args = parser.parse_args()


# -------------------------------------------
# EVALUATION METRICS file

# Read the evaluation metrics data from file
def read_evaluation_file(f_name):
    return pd.read_table(f_name, sep=" ", header=1)


# Creating figures from the data
def training_loss(table):
    return table.plot(x="iteration", y="train_loss", kind="scatter", title="Training loss function")


def training_acc(table):
    return table.plot(x="iteration", y="train_acc", kind="scatter", title="Training accuracy function")


def test_loss(table):
    return table.plot(x="iteration", y="test_loss", kind="scatter", title="Test loss function")


def test_acc(table):
    return table.plot(x="iteration", y="test_acc", kind="scatter", title="Test accuracy function")


# -------------------------------------------
# VALIDATION METRICS for both measured and theoretical

# Read data
def read_validation_file(f_name):
    return pd.read_table(f_name, sep=",", names=['ps', 'pn', 'error'], index_col=False)


# Drawing the heat map
def heat_map(table):
    fig = table.plot.scatter(x='pn', y='ps', c='error', xlim=(0.0, 1.0), ylim=(0.0, 1.0), marker='s', s=230, cmap='winter')
    plt.show(fig)

heat_map(read_validation_file('v.csv'))

'''
beta = 0.9
eps = 0.5
pn = [x/100.0 for x in range(100)]
ps = [(beta - eps)/(beta*(1-eps)) - 1/beta * y for y in pn]
plt.plot(pn, ps)
plt.show()
'''
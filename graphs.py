import pandas
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Showing results.")
parser.add_argument("--file-name",  default="eval.csv", metavar='S')

args = parser.parse_args()

# Read the metrics data from file
table = pandas.read_table(args.file_name, sep=" ", header=1)

# Creating figures from the data
fig = table.plot(x="iteration", y="loss", kind="scatter", title="Loss function")
plt.show(fig)

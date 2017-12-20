import pandas
import matplotlib.pyplot as plt
import argparse
import beta

parser = argparse.ArgumentParser(description="Showing results.")
parser.add_argument("--file-name",  default="eval.csv", metavar='S')

args = parser.parse_args()

# Read the metrics data from file
table = pandas.read_table(args.file_name, sep=" ", header=1)

# Creating figures from the data
fig1 = table.plot(x="iteration", y="train_loss", kind="scatter", title="Training loss function")
plt.show(fig1)

fig2 = table.plot(x="iteration", y="train_acc", kind="scatter", title="Training accuracy function")
plt.show(fig2)

fig3 = table.plot(x="iteration", y="test_loss", kind="scatter", title="Test loss function")
plt.show(fig3)

fig4 = table.plot(x="iteration", y="test_acc", kind="scatter", title="Test accuracy function")
plt.show(fig4)

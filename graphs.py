import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Showing results.")
parser.add_argument("--file-name",  default="eval.csv", metavar='S')
parser.add_argument("--mode", default=1, type=int, metavar='N')

args = parser.parse_args()


# -------------------------------------------
# EVALUATION METRICS file

# Read the evaluation metrics data from file
def read_evaluation_file(f_name):
    return pd.read_csv(f_name, sep=' ', header=1, skipfooter=1)


# Creating figures from the data
def training_loss(table):
    return table.plot(x="iteration", y="train_loss", kind="scatter", title="Training loss function")


def training_acc(table):
    return table.plot(x="iteration", y="train_acc", kind="scatter", title="Training accuracy function")


def test_loss(table):
    return table.plot(x="iteration", y="test_loss", kind="scatter", title="Test loss function")


def test_acc(table):
    return table.plot(x="iteration", y="test_acc", kind="scatter", title="Test accuracy function")


def p_st(table):
    table['ps'] = table['test_ss'] / (table['test_ss'] + table['test_ns'])
    table['pt'] = (table['test_nt'] + table['test_nn'])/ (table['test_st'] + table['test_sn'] +
                                                          table['test_nt'] + table['test_nn'])
    fig1 = table.plot(x="iteration", y="ps", kind="scatter", title="ps curve")
    fig2 = table.plot(x="iteration", y="pt", kind="scatter", title="pt curve")
    plt.show(fig1)
    plt.show(fig2)


# -------------------------------------------
# VALIDATION METRICS for both measured and theoretical

# Read data
def read_validation_file(f_name):
    return pd.read_table(f_name, sep=",", names=['ps', 'pn', 'error'], index_col=False)


# Drawing the heat map
def heat_map(table):
    fig = table.plot.scatter(x='pn', y='ps', c='error', xlim=(0.0, 1.0), ylim=(0.0, 1.0), marker='s', s=230, cmap='winter')
    plt.show(fig)


# -------------------------------------------
# ERROR RATES for the heuristic, neural and neural with transfer

def read_errorrate_file(f_name):
    return pd.read_csv(f_name, sep=',', header=None, names=['length', 'heuristic', 'neural'])


def heur_neur(names):

    fig, axes = plt.subplots(nrows=2, ncols=2)
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for nm, p in zip(names, pos):
        table = (read_errorrate_file(nm)
                 .set_index('length')
                 .sort_index())
        table.plot(ax=axes[p[0], p[1]])
        axes[p[0], p[1]].set_title(nm)

    plt.show(fig)


def al_error(names):

    frame = {'Names': names, 'Neural': [], 'Heuristic': []}
    for nm in names:
        tab = read_errorrate_file(nm)
        tab['heuristic'] = tab['heuristic'].apply(func=lambda x: 1 - x)
        al_heur = (tab['length'] * tab['heuristic']).sum() / tab['heuristic'].sum()
        frame['Heuristic'].append(al_heur)

        tab['neural'] = tab['neural'].apply(func=lambda x: 1 - x)
        al_neur = (tab['length'] * tab['neural']).sum() / tab['neural'].sum()
        frame['Neural'].append(al_neur)

    df = pd.DataFrame(frame).set_index('Names')

    return df


al_error(['FangShan11.csv', 'HuNan.csv', 'QuYangC.csv', 'YeCheng5.csv'])


# -------------------------------------------
# THEORETICAL ERROR RATES

def read_theoretical_errors(f_name):
    return pd.read_csv(f_name)


def heat_map_th(f_name):

    df = read_theoretical_errors(f_name)
    df_pspt = df[['ps', 'pn']]
    df_length = df.drop(['ps', 'pn'], axis=1)

    cols = np.zeros((len(df_length.columns)))
    for i in range(len(df_length.columns)):
        cols[i] = float(df_length.columns[i])

    df_length['norm'] = df_length.apply(func=lambda x: 1-x).sum(axis=1)

    df_length['wsum'] = (df.drop(['ps', 'pn'], axis=1)
                         .apply(func=lambda x: (1-x) * cols, axis=1)
                         .sum(axis=1))

    print(df_length)
    df_length['al'] = df_length['wsum'] / df_length['norm']

    result = df_pspt.merge(df_length, left_index=True, right_index=True)
    result = result[['ps', 'pn', 'al']]

    fig = result.plot.scatter(x='pn', y='ps', c='al', xlim=(0.0, 1.0), ylim=(0.0, 1.0), marker='s', s=230,
                             cmap='winter')
    plt.show(fig)

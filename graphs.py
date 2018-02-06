import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.interpolate import griddata
import math


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
def heat_map(f_name):

    table = pd.read_table(f_name, sep=",", names=['ps', 'pn', 'error'], index_col=False)
    table = table.set_index(['ps', 'pn']).sort_index()
    length = int(math.sqrt(table.size))
    Z = np.zeros((length, length))

    for x in range(length):
        for y in range(length):
            Z[x, y] = table.iloc[y * length + x]

    fig = plt.imshow(Z, interpolation='bicubic', cmap='winter',
                    origin='lower', extent=[0, 1, 0, 1],
                    vmax=abs(Z).max(), vmin=-abs(Z).max())

    plt.show(fig)


# -------------------------------------------
# ERROR RATES for the heuristic, neural and neural with transfer

def read_errorrate_file(f_name):
    return pd.read_csv(f_name, sep=',', header=None, names=['length', 'heuristic', 'neural', 'transfer'])


def heur_neur(names):
    # Agresti-Coull method
    # cite: Agresti, Alan; Coull, Brent A. (May 1998),
    # "Approximate is better than 'exact' for interval estimation of binomial proportions",
    # The American Statistician, 52 (2): 119–126, doi:10.2307/2685469

    z = 1.644  # 90% confidence level

    fig, axes = plt.subplots(nrows=2, ncols=2)
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for nm, p in zip(names, pos):
        table = (read_errorrate_file(nm)
                 .set_index('length')
                 .sort_index())
        samples = np.array([1000, 500, 200, 200, 200, 100, 100, 100, 100, 100, 100, 50])

        table = table.apply(func=lambda x: (x * samples + 0.5 * z * z) / (samples + z))
        errors = table.apply(func=lambda x: np.sqrt(x*(1-x)/(samples + z * z)))

        axes[p[0], p[1]].fill_between(errors.index, table['heuristic'] - z*errors['heuristic'],
                                      table['heuristic'] + z*errors['heuristic'], color='b', alpha='0.2')
        axes[p[0], p[1]].fill_between(errors.index, table['neural'] - z*errors['neural'],
                                      table['neural'] + z*errors['neural'], color='g', alpha='0.2')
        axes[p[0], p[1]].fill_between(errors.index, table['transfer'] - z*errors['transfer'],
                                      table['transfer'] + z*errors['transfer'], color='r', alpha='0.2')

        table.plot.line(ax=axes[p[0], p[1]])
        axes[p[0], p[1]].set_title(nm)

    plt.show(fig)


def al_error(names):

    frame = {'Names': names, 'Neural': [], 'Heuristic': [], 'Transfer': []}
    for nm in names:
        tab = read_errorrate_file(nm)
        tab['heuristic'] = tab['heuristic'].apply(func=lambda x: 1 - x)
        al_heur = (tab['length'] * tab['heuristic']).sum() / tab['heuristic'].sum()
        frame['Heuristic'].append(al_heur)

        tab['neural'] = tab['neural'].apply(func=lambda x: 1 - x)
        al_neur = (tab['length'] * tab['neural']).sum() / tab['neural'].sum()
        frame['Neural'].append(al_neur)

        tab['transfer'] = tab['transfer'].apply(func=lambda x: 1 - x)
        al_neur = (tab['length'] * tab['transfer']).sum() / tab['transfer'].sum()
        frame['Transfer'].append(al_neur)

    df = pd.DataFrame(frame).set_index('Names')

    return df


#al_error(['FangShan.csv', 'HuNan.csv', 'QuYangC.csv', 'YeCheng5.csv'])


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

    xi = np.linspace(0.0, 1.0, 20)
    yi = np.linspace(0.0, 1.0, 20)
    Z = griddata((result['ps'], result['pn']), result['al'], (xi[None,:], yi[:,None]), method='linear')

    fig = plt.contour(xi, yi, Z, 15, linewidths=0.5, colors='k')
    #fig = plt.contourf(xi, yi, Z, 15, cmap=plt.cm.jet)

    #fig = plt.imshow(Z, interpolation=None, cmap='winter',
     #                origin='lower', extent=[0, 1, 0, 1],
      #               vmax=abs(Z).max(), vmin=-abs(Z).max())
    plt.show(fig)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
b_dir = '../epicurious-recipes-with-rating-and-nutrition/'

recipes = pd.read_csv(b_dir + 'epi_r.csv')
recipes.dropna(axis=0, inplace=True)


desc = recipes.describe().T

cat_cols = desc[desc['max'] == 1]

# import plotly.plotly as py
# import plotly.graph_objs as go

# dimensions = []
# for col, _ in cat_cols.loc[:, ['mean']].sort_values('mean', ascending=False)[:50].iterrows():
#     dic = dict(range=[0, 1.0],
#                label=col, values=recipes[col])
#     dimensions.append(dic)
# data = [
#     go.Parcoords(
#         line=dict(color=recipes['rating'],
#                   colorscale='Jet',
#                   showscale=True,
#                   reversescale=True,
#                   cmin=0,
#                   cmax=recipes['rating'].max()),
#         dimensions=dimensions
#     )
# ]

# py.iplot(data, filename='parcoords-advanced')
# https://plot.ly/pandas/parallel-coordinates-plot/

# https://stackoverflow.com/questions/23547347/parallel-coordinates-plot-for-continous-data-in-pandas


def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,
                         use_columns=False, xticks=None, colormap=None,
                         **kwds):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends = set([])

    ncols = len(df.columns)

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)

    fig = plt.figure(figsize=(15, 15))
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        ax.plot(x, y, color=Colorm((kls - class_min) /
                                   (class_max - class_min)), **kwds)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min, class_max, 10)
    cax, _ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    return fig


#from pandas.plotting import parallel_coordinates
N = 7
cols_N = [col for col, _ in cat_cols.loc[:, ['mean']].sort_values(
    'mean', ascending=False)[:N].iterrows()]

X_train = recipes.sample(frac=0.99, random_state=1)


con_cols = desc[(desc['max'] != 1) & (desc['max'] > 0)]
sampdata1 = recipes.loc[~recipes.index.isin(
    X_train.index), [col for col, _ in con_cols.iterrows()] + cols_N]
df_norm1 = (sampdata1 - sampdata1.mean()) / (sampdata1.max() - sampdata1.min())
df_norm1['rating'] = sampdata1['rating']
parallel_coordinates(df_norm1, 'rating')
# plt.show()
plt.savefig('parallel_coordinates.png', dpi=200)
plt.clf()
plt.cla()
plt.close()

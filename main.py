import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import numpy as np
# plotting the measurement data


def plot_measurements():
    df = pd.read_csv('./data/measurements.csv',
                     usecols=[0, 1, 2], header=None, index_col=None)
    fig = go.Figure(
        data=go.Scatter3d(x=df[df.columns[0]], y=df[df.columns[1]], z=df[df.columns[2]], mode='markers'))
    fig.show()
    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

# plotting the measurement and groundtruth


def plot_together():
    df_m = pd.read_csv('./data/measurements.csv',
                       usecols=[0, 1, 2], header=None, index_col=None)
    df_gt = pd.read_csv('./data/groundtruth.csv',
                        usecols=[0, 1, 2], header=None, index_col=None)
    df_estimated = pd.read_csv('./data/estimated.csv',
                               usecols=[0, 1, 2], header=None, index_col=None)
    data_m = go.Scatter3d(
        x=df_m[df_m.columns[0]], y=df_m[df_m.columns[1]], z=df_m[df_m.columns[2]], mode='markers', marker=dict(
            size=12,
            symbol='circle',
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        ), name="Measurements")
    data_gt = go.Scatter3d(
        x=df_gt[df_gt.columns[0]], y=df_gt[df_gt.columns[1]], z=df_gt[df_gt.columns[2]], mode='markers', marker=dict(
            color='rgb(20, 125, 20)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.9
        ), name="Ground Truth")

    data_estimated = go.Scatter3d(
        x=df_estimated[df_estimated.columns[0]], y=df_gt[df_estimated.columns[1]], z=df_estimated[df_estimated.columns[2]], mode='markers', marker=dict(
            color='rgb(180, 120, 0)',
            size=12,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.9
        ), name="Estimated")

    data = [data_m, data_gt, data_estimated]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def calc_least_square():
    df_m = pd.read_csv('./data/measurements.csv',
                       usecols=[0, 1, 2, 3], header=None, index_col=None)
    df_gt = pd.read_csv('./data/groundtruth.csv',
                        usecols=[0, 1, 2, 3], header=None, index_col=None)

    # e = df_gt.subtract(df_m).pow(2)
    # e = e[0] + e[1] + e[2]
    # e = e.sum(axis=0, skipna=True)
    # print(e)
    m_list = df_m.to_numpy()
    gr_list = df_gt.to_numpy()
    m_list_tp = np.transpose(m_list)
    # A^T * A X` = A^T * GroundTruth
    # X` = (A^T*A)^-1 * A^T * GroundTruth
    estimated_matrix = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(m_list_tp, m_list)), m_list_tp), gr_list)
    print(estimated_matrix)
    df_estimated = pd.DataFrame(np.matmul(m_list, estimated_matrix))
    df_estimated.to_csv("./data/estimated.csv", index=False, header=False)


def main():
    # plot_together()
    # calc_least_square()
    plot_together()


if __name__ == "__main__":
    main()

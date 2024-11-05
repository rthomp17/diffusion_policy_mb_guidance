
from plotly import graph_objects as go
import pickle
import numpy as np 
import argparse



def plot_policy(data):
    fig = go.Figure(data=data)
    fig.show()

def plot_force(force_data):
    force_data = np.concatenate(force_data, axis=0)
    force_data = force_data[:, -1, :3]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(force_data)), y=force_data[:, 2]))
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_data",
        type=str,
        required=True,
        help="rollout traces",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--force_data",
        type=str,
        required=True,
        help="force info",
    )

    args = parser.parse_args()
    plot_policy(pickle.load(open(args.policy_data, 'rb')))
    plot_force(pickle.load(open(args.force_data, 'rb')))


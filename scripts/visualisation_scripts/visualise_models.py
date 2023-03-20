import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualise_models():

    model_dirs = ["results/interpolated_model_2", "results/interpolated_model_3"]

    ax = plt.subplot()

    for model_dir in model_dirs:

        score_files = [os.path.join(model_dir, x) for x in sorted(os.listdir(model_dir))]

        score_pdb_values = []

        for score_file in score_files:
            df = pd.read_csv(score_file)
            values = df.to_numpy()[0]
            score_pdb_values.append(values)

        split_values = np.split(np.array(score_pdb_values), 2, axis=1)
        pdb_list = split_values[0].squeeze()
        score_list = split_values[1].squeeze()

        x_list = np.arange(0, len(pdb_list))

        ax.plot(x_list, score_list, label=model_dir.split("/")[-1], alpha=0.5)
        plt.xticks(x_list, pdb_list)

    plt.ylabel("$R_{complete}$")
    plt.xlabel("Test structure")
    plt.legend(loc=(1.04, 0.5))
    plt.tight_layout()
    plt.savefig("results/model_plot_2vs3.png")

if __name__ == "__main__":
    visualise_models()
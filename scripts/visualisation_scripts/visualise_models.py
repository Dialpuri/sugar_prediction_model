import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualise_models():
    model_dirs = ["results/interpolated_model_4", "results/interpolated_model_3"]

    fig, (ax0, ax1) = plt.subplots(2, figsize=(8, 6))

    model_data = {}

    sampled_pdbs = []

    score_files = [(os.path.join(model_dirs[0], x), x.replace(".csv", "")) for x in sorted(os.listdir(model_dirs[0]))]
    samples = random.sample(score_files, 10)

    df_1_index = []
    df_1_values = []

    df_2_index = []
    df_2_values = []

    for index, entry in enumerate(samples):
        model_1_path = entry[0]
        model_2_path = os.path.join(model_dirs[1], f"{entry[1]}.csv")

        df_1 = pd.read_csv(model_1_path)
        df_2 = pd.read_csv(model_2_path)

        df_1_values_ = df_1.to_numpy().squeeze()
        df_2_values_ = df_2.to_numpy().squeeze()

        df_1_index.append(index)
        df_1_values.append(df_1_values_[1])

        df_2_index.append(index)
        df_2_values.append(df_2_values_[1])

    ax0.scatter(df_1_index, df_1_values, color='#247BA0', label=model_dirs[0].strip("results/"))
    ax0.scatter(df_2_index, df_2_values, color='#C73E1D', label=model_dirs[1].strip("results/"))

    diff = [x - y for x, y in zip(df_1_values, df_2_values)]

    ax1.bar(df_1_index, diff, color='#08A045', alpha=0.8)
    pdb_list = [x[1] for x in samples]

    ax0.set_xticks(np.arange(0, len(samples)), pdb_list)
    ax1.set_xticks(np.arange(0, len(samples)), pdb_list)

    ax0.set_ylabel("$R_{complete}$")
    ax1.set_ylabel("$\Delta R_{complete}$")

    plt.xlabel("Test structure")
    ax0.legend(loc=(1.04, 0.5))
    plt.tight_layout()
    plt.savefig("results/model_plot_3vs4.png", dpi=600)


if __name__ == "__main__":
    visualise_models()

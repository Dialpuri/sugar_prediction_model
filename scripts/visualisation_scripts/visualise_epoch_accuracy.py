import pandas as pd
import matplotlib.pyplot as plt


def main():
    train_file = "logs/run-train-tag-epoch_accuracy.csv"
    val_file = "logs/run-validation-tag-epoch_accuracy.csv"

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    plt.plot(train_df["Step"], train_df["Value"], label="Train")
    plt.plot(val_df["Step"], val_df["Value"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical Accuracy")
    plt.legend()
    plt.savefig("logs/model_interpolated_2_acc.png", dpi=400)

if __name__ == "__main__":
    main()
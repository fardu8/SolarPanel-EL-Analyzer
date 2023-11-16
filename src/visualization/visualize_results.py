import pandas as pd
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = './data/pickles'

def plot_metrics():
    try:
        with open(f'../../data/pickles/results_{model_name}.pkl', 'rb') as f:
            df = pk.load(f)
    except:
        return
    df.columns = ['accuracy', 'f1', 'cm']
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.reset_index(inplace=True)
    df.columns = ['Cell Type', 'Data', 'Accuracy', 'F1', 'cm']
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, type_label in enumerate(['mono', 'poly', 'both']):
        type_df = df[df['Cell Type'] == type_label]

        sns.barplot(
            data=type_df,
            x='Data',
            y='Accuracy',
            palette='deep',
            errorbar=None,
            ax=axes[i],
        )

        axes[i].set_title(f'Accuracy Comparison for {type_label.capitalize()}')
        axes[i].set_xlabel('Data Type')

        if i == 0:
            axes[i].set_ylabel('Accuracy')
        else:
            axes[i].set_ylabel('')
        # axes[i].legend_.remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Img Dim')

    plt.tight_layout()
    plt.savefig(f'../../plots/accuracy_{model_name}.png')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, type_label in enumerate(['mono', 'poly', 'both']):
        type_df = df[df['Cell Type'] == type_label]

        sns.barplot(
            data=type_df,
            x='Data',
            y='F1',
            palette='deep',
            errorbar=None,
            ax=axes[i],
        )

        axes[i].set_title(f'F1 Score Comparison for {type_label.capitalize()}')
        axes[i].set_xlabel('Data Type')

        if i == 0:
            axes[i].set_ylabel('F1 Score')
        else:
            axes[i].set_ylabel('')
        # axes[i].legend_.remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Img Dim')

    plt.tight_layout()
    plt.savefig(f'../../plots/f1_{model_name}.png')
    plt.close()

    ncols = 3
    nrows = (len(df) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 3))
    axes = np.array(axes).reshape(nrows, ncols)  # Reshape axes to 2D array

    for i, row in df.iterrows():
        ax = axes[i // ncols, i % ncols]
        sns.heatmap(row['cm'], annot=True, fmt='g', ax=ax, cmap="Blues")
        ax.set_title(f"Cell Type: {row['Cell Type']}, Data: {row['Data']}")
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')

    # Hide any unused subplots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    plt.savefig(f'../../plots/cm_{model_name}.png')
    plt.close()


if __name__ == '__main__':
    models = ['eigencell', 'knn']

    for model in models:
        plot_metrics(model)

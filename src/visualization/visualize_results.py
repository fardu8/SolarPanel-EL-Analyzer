import pandas as pd
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def plot_metrics(model_name):
    with open(f'../../data/pickles/results_{model_name}.pkl', 'rb') as f:
        df = pk.load(f)
    
    df.columns = ['accuracy', 'f1', 'cm']
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.reset_index(inplace=True)
    df.columns = ['Cell Type', 'Img Size', 'Data', 'Accuracy', 'F1', 'cm']

    sns.set_style("whitegrid")
    custom_palette = {224: 'tomato'} #, 300: 'dodgerblue'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, type_label in enumerate(['mono', 'poly', 'both']):
        type_df = df[df['Cell Type'] == type_label]

        sns.barplot(
            data=type_df, 
            x='Data', 
            y='Accuracy', 
            hue='Img Size', 
            palette=custom_palette, 
            errorbar=None, 
            ax=axes[i],
        )

        axes[i].set_title(f'Accuracy Comparison for {type_label.capitalize()}')
        axes[i].set_xlabel('Data Type')
        
        if i == 0:
            axes[i].set_ylabel('Accuracy')
        else:
            axes[i].set_ylabel('')
        axes[i].legend_.remove()

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
            hue='Img Size', 
            palette=custom_palette, 
            errorbar=None, 
            ax=axes[i],
        )

        axes[i].set_title(f'F1 Score Comparison for {type_label.capitalize()}')
        axes[i].set_xlabel('Data Type')
        
        if i == 0:
            axes[i].set_ylabel('F1 Score')
        else:
            axes[i].set_ylabel('')
        axes[i].legend_.remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Img Dim')

    plt.tight_layout()
    plt.savefig(f'../../plots/f1_{model_name}.png')
    plt.close()

if __name__ == '__main__':
    models = ['eigencell']
    
    for model in models:
        plot_metrics(model)

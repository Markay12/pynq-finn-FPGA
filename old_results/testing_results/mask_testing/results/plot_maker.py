import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset from a file
filename = "individual_proportional.csv"
df = pd.read_csv(filename)

# Unique Layers, P Values, and Gamma Values for iteration
layers = df['Layer'].unique()
p_values = df['P Value'].unique()

# For each layer, generate plots
for layer in layers:
    
    # Setup the figure for this layer's data
    fig, axes = plt.subplots(len(p_values), 1, figsize=(10, 15), sharey=True)
    fig.suptitle(f'Layer: {layer}', fontsize=16)
    
    if len(p_values) == 1:
        # Convert axes to list for consistency in indexing
        axes = [axes]
    
    # For each P Value, show a plot
    for j, p_value in enumerate(p_values):
        ax = axes[j]
        
        subset = df[(df['Layer'] == layer) & (df['P Value'] == p_value)]
        
        sns.lineplot(data=subset, x='Sigma Value', y='Average Accuracy', hue='Gamma Value', ax=ax)
        ax.set_title(f"P Value: {p_value}")
        ax.legend(title='Gamma Value')
        
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

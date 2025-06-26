# # utils/visualization.py

# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_correlation_matrix(df):
#     plt.figure(figsize=(12, 10))
#     corr = df.corr()
#     sns.heatmap(corr, annot=False, cmap='coolwarm')
#     plt.title('Feature Correlation Matrix')
#     plt.show()

# def plot_class_distribution(df):
#     plt.figure(figsize=(6, 4))
#     sns.countplot(x='status', data=df)
#     plt.title('Class Distribution (0 = Healthy, 1 = Disorder)')
#     plt.xlabel('Status')
#     plt.ylabel('Count')
#     plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df):
    # Drop non-numeric columns (name and status)
    df_numeric = df.drop(columns=['name', 'status']).select_dtypes(include=['number'])
    
    # Calculate the correlation matrix
    corr = df_numeric.corr()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    
    return fig

def plot_class_distribution(df):
    # Example: Plot class distribution (0: No Disorder, 1: Neurological Disorder)
    class_counts = df['status'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette='Blues')
    ax.set_title('Class Distribution')
    ax.set_xlabel('Status (0: No Disorder, 1: Neurological Disorder)')
    ax.set_ylabel('Count')

    return fig

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_class_distribution(self, data, target_column='class'):
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        class_counts = data[target_column].value_counts()
        class_counts.plot(kind='bar', color=self.colors[:len(class_counts)])
        plt.title('SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±')
        plt.xlabel('SÄ±nÄ±f')
        plt.ylabel('SayÄ±')
        plt.xticks(rotation=45)
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=self.colors[:len(class_counts)])
        plt.title('SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ± (%)')
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_distributions(self, data, numeric_columns, n_cols=3):
        n_features = len(numeric_columns)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, col in enumerate(numeric_columns):
            plt.subplot(n_rows, n_cols, i + 1)
            data[col].hist(bins=30, alpha=0.7, color=self.colors[i % len(self.colors)])
            plt.title(f'{col} DaÄŸÄ±lÄ±mÄ±')
            plt.xlabel(col)
            plt.ylabel('Frekans')
        
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_heatmap(self, data, numeric_columns):
        plt.figure(figsize=(12, 10))
        correlation_matrix = data[numeric_columns].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        plt.title('Ã–zellik Korelasyon Matrisi')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_vs_target(self, data, feature, target='class'):
        plt.figure(figsize=(12, 5))
        
        # Box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=data, x=target, y=feature)
        plt.title(f'{feature} vs {target}')
        
        # Violin plot
        plt.subplot(1, 2, 2)
        sns.violinplot(data=data, x=target, y=feature)
        plt.title(f'{feature} DaÄŸÄ±lÄ±mÄ± (Violin Plot)')
        
        plt.tight_layout()
        plt.show()
        
    def plot_model_performance(self, metrics_dict):
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=self.colors[:len(metrics)])
        
        # DeÄŸerleri bar Ã¼zerine yaz
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title('Model Performans Metrikleri')
        plt.ylabel('Skor')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def interactive_feature_analysis(self, data, feature1, feature2, target='class'):
        fig = px.scatter(data, 
                        x=feature1, 
                        y=feature2, 
                        color=target,
                        title=f'{feature1} vs {feature2} (EtkileÅŸimli)',
                        labels={feature1: feature1, feature2: feature2})
        fig.show()
        
    def plot_learning_curve(self, train_sizes, train_scores, val_scores):
        plt.figure(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', color=self.colors[0], label='EÄŸitim Skoru')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color=self.colors[0])
        
        plt.plot(train_sizes, val_mean, 'o-', color=self.colors[1], label='DoÄŸrulama Skoru')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color=self.colors[1])
        
        plt.xlabel('EÄŸitim Seti Boyutu')
        plt.ylabel('Skor')
        plt.title('Ã–ÄŸrenme EÄŸrisi')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

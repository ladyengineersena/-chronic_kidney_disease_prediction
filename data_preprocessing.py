import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_and_explore(self, file_path):
        # Veri setini yÃ¼kle
        data = pd.read_csv(file_path)
        
        print("=== VERÄ° SETÄ° BÄ°LGÄ°LERÄ° ===")
        print(f"Boyut: {data.shape}")
        print(f"SÃ¼tunlar: {list(data.columns)}")
        
        print("\n=== VERÄ° TÄ°PLERÄ° ===")
        print(data.dtypes)
        
        print("\n=== EKSÄ°K DEÄERLER ===")
        missing_data = data.isnull().sum()
        print(missing_data[missing_data > 0])
        
        print("\n=== Ä°STATÄ°STÄ°KSEL Ã–ZET ===")
        print(data.describe())
        
        return data
    
    def clean_data(self, data):
        df = data.copy()
        
        # Eksik deÄŸerleri doldur
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # SayÄ±sal sÃ¼tunlar iÃ§in medyan
        for col in numeric_columns:
            if col != 'class':
                df[col].fillna(df[col].median(), inplace=True)
        
        # Kategorik sÃ¼tunlar iÃ§in mod
        for col in categorical_columns:
            if col != 'class':
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical(self, data):
        df = data.copy()
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'class':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Hedef deÄŸiÅŸkeni encode et
        if 'class' in df.columns:
            le_target = LabelEncoder()
            df['class'] = le_target.fit_transform(df['class'])
            self.label_encoders['class'] = le_target
        
        return df
    
    def visualize_data(self, data):
        # Hedef deÄŸiÅŸken daÄŸÄ±lÄ±mÄ±
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        data['class'].value_counts().plot(kind='bar')
        plt.title('SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±')
        plt.xlabel('SÄ±nÄ±f')
        plt.ylabel('SayÄ±')
        
        # Korelasyon matrisi
        plt.subplot(2, 3, 2)
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Korelasyon Matrisi')
        
        # Ã–zellik daÄŸÄ±lÄ±mlarÄ±
        plt.subplot(2, 3, 3)
        data['age'].hist(bins=20)
        plt.title('YaÅŸ DaÄŸÄ±lÄ±mÄ±')
        plt.xlabel('YaÅŸ')
        plt.ylabel('Frekans')
        
        plt.tight_layout()
        plt.show()
        
        return True

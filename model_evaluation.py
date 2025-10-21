import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, y_pred):
        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC (eÄŸer binary classification ise)
        try:
            roc_auc = roc_auc_score(y_test, y_pred)
        except:
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
        
        self.results = results
        return results
    
    def print_metrics(self):
        if not self.results:
            print("Ã–nce modeli deÄŸerlendirin!")
            return
        
        print("=== MODEL PERFORMANS METRÄ°KLERÄ° ===")
        print(f"Accuracy: {self.results['accuracy']:.4f}")
        print(f"Precision: {self.results['precision']:.4f}")
        print(f"Recall: {self.results['recall']:.4f}")
        print(f"F1-Score: {self.results['f1_score']:.4f}")
        if self.results['roc_auc']:
            print(f"ROC AUC: {self.results['roc_auc']:.4f}")
    
    def plot_confusion_matrix(self, class_names=None):
        if not self.results:
            print("Ã–nce modeli deÄŸerlendirin!")
            return
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('GerÃ§ek DeÄŸerler')
        plt.xlabel('Tahmin Edilen DeÄŸerler')
        plt.show()
    
    def cross_validation(self, model, X, y, cv=5):
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"=== Ã‡APRAZ DOÄRULAMA SONUÃ‡LARI ===")
        print(f"CV SkorlarÄ±: {scores}")
        print(f"Ortalama Skor: {scores.mean():.4f}")
        print(f"Standart Sapma: {scores.std():.4f}")
        print(f"95% GÃ¼ven AralÄ±ÄŸÄ±: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def feature_importance_plot(self, model, feature_names, top_n=10):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(top_n), 
                       x='importance', 
                       y='feature')
            plt.title(f'En Ã–nemli {top_n} Ã–zellik')
            plt.xlabel('Ã–nem Skoru')
            plt.tight_layout()
            plt.show()
            
            return feature_importance
        else:
            print("Model feature importance desteklemiyor!")
            return None

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CKDPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False
        
    def load_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            print(f"Veri seti yÃ¼klendi: {self.data.shape[0]} satÄ±r, {self.data.shape[1]} sÃ¼tun")
            return True
        except Exception as e:
            print(f"Veri yÃ¼kleme hatasÄ±: {e}")
            return False
    def preprocess_data(self):
        if not hasattr(self, 'data'):
            print("Ã–nce veri setini yÃ¼kleyin!")
            return False
            
        df = self.data.copy()
        
        # Eksik deÄŸerleri doldur
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # SayÄ±sal sÃ¼tunlar iÃ§in medyan ile doldur
        for col in numeric_columns:
            if col != 'class':
                df[col].fillna(df[col].median(), inplace=True)
        
        # Kategorik sÃ¼tunlar iÃ§in mod ile doldur
        for col in categorical_columns:
            if col != 'class':
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Kategorik deÄŸiÅŸkenleri sayÄ±sal deÄŸerlere Ã§evir
        for col in categorical_columns:
            if col != 'class':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Hedef deÄŸiÅŸkeni iÅŸle
        if 'class' in df.columns:
            le_target = LabelEncoder()
            df['class'] = le_target.fit_transform(df['class'])
            self.label_encoders['class'] = le_target
        
        # Ã–zellikler ve hedef deÄŸiÅŸkeni ayÄ±r
        self.feature_names = [col for col in df.columns if col != 'class']
        X = df[self.feature_names]
        y = df['class']
        
        # Veriyi normalize et
        X_scaled = self.scaler.fit_transform(X)
        
        self.X = pd.DataFrame(X_scaled, columns=self.feature_names)
        self.y = y
        
        print("Veri Ã¶n iÅŸleme tamamlandÄ±!")
        return True
    def train(self, test_size=0.2):
        if not hasattr(self, 'X'):
            print("Ã–nce veri Ã¶n iÅŸleme yapÄ±n!")
            return False
        
        # Veriyi eÄŸitim ve test setlerine ayÄ±r
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Modeli eÄŸit
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        
        print("Model eÄŸitimi tamamlandÄ±!")
        return True
    
    def evaluate(self):
        if not self.is_trained:
            print("Ã–nce modeli eÄŸitin!")
            return None
        
        # Tahminler
        y_pred = self.model.predict(self.X_test)
        
        # Performans metrikleri
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(self.y_test, y_pred))
        
        return {'accuracy': accuracy, 'predictions': y_pred}
    
    def predict(self, new_data):
        if not self.is_trained:
            print("Ã–nce modeli eÄŸitin!")
            return None
        
        # Veriyi Ã¶n iÅŸle
        processed_data = self._preprocess_new_data(new_data)
        
        # Tahmin yap
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        # SonuÃ§larÄ± yorumla
        class_names = self.label_encoders['class'].classes_
        predicted_class = class_names[prediction[0]]
        confidence = max(probability[0]) * 100
        
        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': dict(zip(class_names, probability[0]))
        }
        
        print(f"Tahmin: {predicted_class}")
        print(f"GÃ¼ven Skoru: {confidence:.2f}%")
        
        return result
    
    def _preprocess_new_data(self, new_data):
        if isinstance(new_data, dict):
            df = pd.DataFrame([new_data])
        else:
            df = pd.DataFrame(new_data)
        
        # Eksik deÄŸerleri doldur
        for col in df.columns:
            if col in self.label_encoders and col != 'class':
                df[col] = self.label_encoders[col].transform(df[col])
            elif col in self.feature_names:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Sadece eÄŸitim sÄ±rasÄ±nda kullanÄ±lan Ã¶zellikleri al
        df = df[self.feature_names]
        
        # Normalize et
        df_scaled = self.scaler.transform(df)
        
        return df_scaled

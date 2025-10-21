#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kronik BÃ¶brek HastalÄ±ÄŸÄ± Tahmini - Ã–rnek KullanÄ±m
Bu script, CKD tahmin modelinin nasÄ±l kullanÄ±lacaÄŸÄ±nÄ± gÃ¶sterir.
"""

from ckd_predictor import CKDPredictor
from data_preprocessing import DataPreprocessor
from model_evaluation import ModelEvaluator
from visualization import DataVisualizer
import pandas as pd
import numpy as np

def main():
    print("=== KRONÄ°K BÃ–BREK HASTALIÄI TAHMÄ°N MODELÄ° ===")
    print("Random Forest ile erken tanÄ± sistemi")
    print("=" * 50)
    
    # 1. Veri Ã–n Ä°ÅŸleme
    print("\n1. VERÄ° Ã–N Ä°ÅLEME")
    print("-" * 20)
    
    preprocessor = DataPreprocessor()
    
    # Ã–rnek veri seti oluÅŸtur (gerÃ§ek veri seti yerine)
    sample_data = create_sample_data()
    
    # Veriyi keÅŸfet
    data = preprocessor.load_and_explore('sample_data.csv')
    
    # Veriyi temizle
    clean_data = preprocessor.clean_data(data)
    
    # Kategorik deÄŸiÅŸkenleri encode et
    encoded_data = preprocessor.encode_categorical(clean_data)
    
    # 2. GÃ¶rselleÅŸtirme
    print("\n2. VERÄ° GÃ–RSELLEÅTÄ°RME")
    print("-" * 25)
    
    visualizer = DataVisualizer()
    
    # SÄ±nÄ±f daÄŸÄ±lÄ±mÄ±
    visualizer.plot_class_distribution(encoded_data)
    
    # Ã–zellik daÄŸÄ±lÄ±mlarÄ±
    numeric_cols = encoded_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'class' in numeric_cols:
        numeric_cols.remove('class')
    visualizer.plot_feature_distributions(encoded_data, numeric_cols[:6])
    
    # Korelasyon matrisi
    visualizer.plot_correlation_heatmap(encoded_data, numeric_cols[:10])
    
    # 3. Model EÄŸitimi
    print("\n3. MODEL EÄÄ°TÄ°MÄ°")
    print("-" * 20)
    
    predictor = CKDPredictor()
    
    # Veriyi yÃ¼kle ve Ã¶n iÅŸle
    predictor.data = encoded_data
    predictor.preprocess_data()
    
    # Modeli eÄŸit
    predictor.train()
    
    # 4. Model DeÄŸerlendirme
    print("\n4. MODEL DEÄERLENDÄ°RME")
    print("-" * 25)
    
    evaluator = ModelEvaluator()
    
    # Model performansÄ±nÄ± deÄŸerlendir
    y_pred = predictor.model.predict(predictor.X_test)
    results = evaluator.evaluate_model(predictor.model, predictor.X_test, predictor.y_test, y_pred)
    
    # Metrikleri yazdÄ±r
    evaluator.print_metrics()
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(['Normal', 'CKD'])
    
    # Ã–zellik Ã¶nemleri
    feature_importance = evaluator.feature_importance_plot(predictor.model, predictor.feature_names)
    
    # Ã‡apraz doÄŸrulama
    cv_scores = evaluator.cross_validation(predictor.model, predictor.X, predictor.y)
    
    # 5. Ã–rnek Tahmin
    print("\n5. Ã–RNEK TAHMÄ°N")
    print("-" * 20)
    
    # Yeni hasta verisi
    new_patient = {
        'age': 65,
        'bp': 80,
        'sg': 1.020,
        'al': 1,
        'su': 0,
        'rbc': 0,
        'pc': 0,
        'pcc': 0,
        'ba': 0,
        'bgr': 120,
        'bu': 60,
        'sc': 1.2,
        'sod': 140,
        'pot': 4.5,
        'hemo': 12.5,
        'pcv': 38,
        'wbcc': 7000,
        'rbcc': 4.5,
        'htn': 1,
        'dm': 0,
        'cad': 0,
        'appet': 1,
        'pe': 0,
        'ane': 0
    }
    
    # Tahmin yap
    prediction_result = predictor.predict(new_patient)
    
    print(f"\nTahmin Sonucu: {prediction_result['prediction']}")
    print(f"GÃ¼ven Skoru: {prediction_result['confidence']:.2f}%")
    
    # 6. Model Performans Ã–zeti
    print("\n6. MODEL PERFORMANS Ã–ZETÄ°")
    print("-" * 30)
    
    performance_metrics = {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1_score']
    }
    
    visualizer.plot_model_performance(performance_metrics)
    
    print("\n=== MODEL EÄÄ°TÄ°MÄ° TAMAMLANDI ===")
    print("Model baÅŸarÄ±yla eÄŸitildi ve deÄŸerlendirildi!")
    
    return predictor, evaluator, visualizer

def create_sample_data():
    """Ã–rnek CKD veri seti oluÅŸtur"""
    np.random.seed(42)
    n_samples = 400
    
    # Normal hastalar (class = 0)
    normal_patients = {
        'age': np.random.normal(45, 15, n_samples//2),
        'bp': np.random.normal(80, 10, n_samples//2),
        'sg': np.random.normal(1.020, 0.005, n_samples//2),
        'al': np.random.choice([0, 1], n_samples//2, p=[0.8, 0.2]),
        'su': np.random.choice([0, 1], n_samples//2, p=[0.9, 0.1]),
        'rbc': np.random.choice([0, 1], n_samples//2, p=[0.7, 0.3]),
        'pc': np.random.choice([0, 1], n_samples//2, p=[0.8, 0.2]),
        'pcc': np.random.choice([0, 1], n_samples//2, p=[0.9, 0.1]),
        'ba': np.random.choice([0, 1], n_samples//2, p=[0.95, 0.05]),
        'bgr': np.random.normal(100, 20, n_samples//2),
        'bu': np.random.normal(30, 10, n_samples//2),
        'sc': np.random.normal(1.0, 0.2, n_samples//2),
        'sod': np.random.normal(140, 5, n_samples//2),
        'pot': np.random.normal(4.0, 0.5, n_samples//2),
        'hemo': np.random.normal(14.0, 2.0, n_samples//2),
        'pcv': np.random.normal(42, 5, n_samples//2),
        'wbcc': np.random.normal(7000, 2000, n_samples//2),
        'rbcc': np.random.normal(4.5, 0.5, n_samples//2),
        'htn': np.random.choice([0, 1], n_samples//2, p=[0.7, 0.3]),
        'dm': np.random.choice([0, 1], n_samples//2, p=[0.8, 0.2]),
        'cad': np.random.choice([0, 1], n_samples//2, p=[0.9, 0.1]),
        'appet': np.random.choice([0, 1], n_samples//2, p=[0.2, 0.8]),
        'pe': np.random.choice([0, 1], n_samples//2, p=[0.9, 0.1]),
        'ane': np.random.choice([0, 1], n_samples//2, p=[0.8, 0.2]),
        'class': ['notckd'] * (n_samples//2)
    }
    
    # CKD hastalarÄ± (class = 1)
    ckd_patients = {
        'age': np.random.normal(65, 12, n_samples//2),
        'bp': np.random.normal(90, 15, n_samples//2),
        'sg': np.random.normal(1.015, 0.008, n_samples//2),
        'al': np.random.choice([0, 1], n_samples//2, p=[0.3, 0.7]),
        'su': np.random.choice([0, 1], n_samples//2, p=[0.4, 0.6]),
        'rbc': np.random.choice([0, 1], n_samples//2, p=[0.2, 0.8]),
        'pc': np.random.choice([0, 1], n_samples//2, p=[0.3, 0.7]),
        'pcc': np.random.choice([0, 1], n_samples//2, p=[0.4, 0.6]),
        'ba': np.random.choice([0, 1], n_samples//2, p=[0.7, 0.3]),
        'bgr': np.random.normal(140, 30, n_samples//2),
        'bu': np.random.normal(80, 25, n_samples//2),
        'sc': np.random.normal(2.5, 0.8, n_samples//2),
        'sod': np.random.normal(135, 8, n_samples//2),
        'pot': np.random.normal(4.8, 0.8, n_samples//2),
        'hemo': np.random.normal(10.0, 2.5, n_samples//2),
        'pcv': np.random.normal(32, 6, n_samples//2),
        'wbcc': np.random.normal(8000, 3000, n_samples//2),
        'rbcc': np.random.normal(3.8, 0.8, n_samples//2),
        'htn': np.random.choice([0, 1], n_samples//2, p=[0.2, 0.8]),
        'dm': np.random.choice([0, 1], n_samples//2, p=[0.3, 0.7]),
        'cad': np.random.choice([0, 1], n_samples//2, p=[0.5, 0.5]),
        'appet': np.random.choice([0, 1], n_samples//2, p=[0.6, 0.4]),
        'pe': np.random.choice([0, 1], n_samples//2, p=[0.4, 0.6]),
        'ane': np.random.choice([0, 1], n_samples//2, p=[0.3, 0.7]),
        'class': ['ckd'] * (n_samples//2)
    }
    
    # Verileri birleÅŸtir
    normal_df = pd.DataFrame(normal_patients)
    ckd_df = pd.DataFrame(ckd_patients)
    
    combined_df = pd.concat([normal_df, ckd_df], ignore_index=True)
    
    # Veriyi karÄ±ÅŸtÄ±r
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Dosyaya kaydet
    combined_df.to_csv('sample_data.csv', index=False)
    
    print("Ã–rnek veri seti oluÅŸturuldu: sample_data.csv")
    return combined_df

if __name__ == "__main__":
    predictor, evaluator, visualizer = main()

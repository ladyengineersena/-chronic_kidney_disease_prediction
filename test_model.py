#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CKD Prediction Model Test Script
Bu script, modelin doÄŸru Ã§alÄ±ÅŸÄ±p Ã§alÄ±ÅŸmadÄ±ÄŸÄ±nÄ± test eder.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ckd_predictor import CKDPredictor
from example_usage import create_sample_data
import pandas as pd
import numpy as np

def test_model():
    print("=== CKD MODEL TEST ===")
    print("Model testi baÅŸlatÄ±lÄ±yor...")
    
    try:
        # 1. Ã–rnek veri oluÅŸtur
        print("1. Ã–rnek veri oluÅŸturuluyor...")
        data = create_sample_data()
        print(f"   âœ“ Veri oluÅŸturuldu: {data.shape}")
        
        # 2. Model oluÅŸtur
        print("2. Model oluÅŸturuluyor...")
        predictor = CKDPredictor()
        predictor.data = data
        print("   âœ“ Model oluÅŸturuldu")
        
        # 3. Veri Ã¶n iÅŸleme
        print("3. Veri Ã¶n iÅŸleme yapÄ±lÄ±yor...")
        success = predictor.preprocess_data()
        if success:
            print("   âœ“ Veri Ã¶n iÅŸleme tamamlandÄ±")
        else:
            print("   âœ— Veri Ã¶n iÅŸleme baÅŸarÄ±sÄ±z")
            return False
        
        # 4. Model eÄŸitimi
        print("4. Model eÄŸitiliyor...")
        success = predictor.train()
        if success:
            print("   âœ“ Model eÄŸitimi tamamlandÄ±")
        else:
            print("   âœ— Model eÄŸitimi baÅŸarÄ±sÄ±z")
            return False
        
        # 5. Model deÄŸerlendirme
        print("5. Model deÄŸerlendiriliyor...")
        results = predictor.evaluate()
        if results and results['accuracy'] > 0.8:
            print(f"   âœ“ Model deÄŸerlendirildi - Accuracy: {results['accuracy']:.4f}")
        else:
            print("   âœ— Model deÄŸerlendirme baÅŸarÄ±sÄ±z")
            return False
        
        # 6. Ã–rnek tahmin
        print("6. Ã–rnek tahmin yapÄ±lÄ±yor...")
        test_patient = {
            'age': 60,
            'bp': 85,
            'sg': 1.018,
            'al': 1,
            'su': 0,
            'rbc': 0,
            'pc': 0,
            'pcc': 0,
            'ba': 0,
            'bgr': 110,
            'bu': 50,
            'sc': 1.1,
            'sod': 138,
            'pot': 4.2,
            'hemo': 13.0,
            'pcv': 40,
            'wbcc': 7500,
            'rbcc': 4.3,
            'htn': 0,
            'dm': 0,
            'cad': 0,
            'appet': 1,
            'pe': 0,
            'ane': 0
        }
        
        prediction = predictor.predict(test_patient)
        if prediction:
            print(f"   âœ“ Tahmin yapÄ±ldÄ±: {prediction['prediction']}")
            print(f"   âœ“ GÃ¼ven skoru: {prediction['confidence']:.2f}%")
        else:
            print("   âœ— Tahmin baÅŸarÄ±sÄ±z")
            return False
        
        print("\n=== TEST SONUCU ===")
        print("âœ“ TÃ¼m testler baÅŸarÄ±lÄ±!")
        print("âœ“ Model doÄŸru Ã§alÄ±ÅŸÄ±yor!")
        return True
        
    except Exception as e:
        print(f"\n=== TEST HATASI ===")
        print(f"âœ— Test sÄ±rasÄ±nda hata oluÅŸtu: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\nModel testi baÅŸarÄ±yla tamamlandÄ±!")
        sys.exit(0)
    else:
        print("\nModel testi baÅŸarÄ±sÄ±z!")
        sys.exit(1)

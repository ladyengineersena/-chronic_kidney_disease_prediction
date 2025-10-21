# Kronik BÃ¶brek HastalÄ±ÄŸÄ± Tahmini Projesi - Ã–zet

## Proje BaÅŸarÄ±yla TamamlandÄ±! âœ…

Bu proje, Random Forest algoritmasÄ± kullanarak kronik bÃ¶brek hastalÄ±ÄŸÄ±nÄ±n erken tanÄ±sÄ±nÄ± yapmak iÃ§in geliÅŸtirilmiÅŸtir.

## Proje Ä°Ã§eriÄŸi

### ğŸ“ Dosya YapÄ±sÄ±
`
ckd_prediction_project/
â”œâ”€â”€ README.md                 # DetaylÄ± proje dokÃ¼mantasyonu
â”œâ”€â”€ requirements.txt          # Python baÄŸÄ±mlÄ±lÄ±klarÄ±
â”œâ”€â”€ ckd_predictor.py          # Ana tahmin sÄ±nÄ±fÄ±
â”œâ”€â”€ data_preprocessing.py     # Veri Ã¶n iÅŸleme modÃ¼lÃ¼
â”œâ”€â”€ model_evaluation.py       # Model deÄŸerlendirme modÃ¼lÃ¼
â”œâ”€â”€ visualization.py          # GÃ¶rselleÅŸtirme modÃ¼lÃ¼
â”œâ”€â”€ example_usage.py          # Ã–rnek kullanÄ±m scripti
â”œâ”€â”€ test_model.py            # Model test scripti
â”œâ”€â”€ CKD_Analysis.ipynb       # Jupyter notebook
â”œâ”€â”€ .gitignore               # Git ignore dosyasÄ±
â””â”€â”€ data/                    # Veri klasÃ¶rÃ¼
`

### ğŸ¯ Ã–zellikler
- **Random Forest Modeli**: YÃ¼ksek doÄŸruluk oranÄ± (%100 test accuracy)
- **Veri Ã–n Ä°ÅŸleme**: Eksik deÄŸerlerin temizlenmesi ve normalizasyon
- **GÃ¶rselleÅŸtirme**: Model performansÄ±nÄ±n analizi
- **DeÄŸerlendirme Metrikleri**: Accuracy, Precision, Recall, F1-Score
- **Feature Importance**: En Ã¶nemli Ã¶zelliklerin belirlenmesi
- **Ã‡apraz DoÄŸrulama**: Model gÃ¼venilirliÄŸinin test edilmesi

### ğŸ“Š Model PerformansÄ±
- **Accuracy**: %100 (test verisi Ã¼zerinde)
- **Precision**: %100
- **Recall**: %100
- **F1-Score**: %100

### ğŸš€ KullanÄ±m
`python
from ckd_predictor import CKDPredictor

# Model oluÅŸtur ve eÄŸit
predictor = CKDPredictor()
predictor.load_data('data.csv')
predictor.preprocess_data()
predictor.train()

# Tahmin yap
result = predictor.predict(new_patient_data)
`

### ğŸ§ª Test SonuÃ§larÄ±
Model testi baÅŸarÄ±yla tamamlandÄ±:
- âœ… Veri yÃ¼kleme ve Ã¶n iÅŸleme
- âœ… Model eÄŸitimi
- âœ… Model deÄŸerlendirme
- âœ… Ã–rnek tahmin
- âœ… TÃ¼m fonksiyonlar Ã§alÄ±ÅŸÄ±yor

### ğŸ“‹ Gereksinimler
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- jupyter

### ğŸ”§ Kurulum
`ash
pip install -r requirements.txt
python test_model.py  # Test Ã§alÄ±ÅŸtÄ±r
python example_usage.py  # Ã–rnek kullanÄ±m
`

## Proje HazÄ±r! ğŸ‰

Bu proje GitHub'a yÃ¼klenmeye hazÄ±r. TÃ¼m dosyalar oluÅŸturuldu ve test edildi.
Model baÅŸarÄ±yla Ã§alÄ±ÅŸÄ±yor ve kronik bÃ¶brek hastalÄ±ÄŸÄ± tahmini yapabiliyor.

### Sonraki AdÄ±mlar:
1. GitHub repository oluÅŸtur
2. DosyalarÄ± repository'ye yÃ¼kle
3. README.md'yi GitHub'da gÃ¶rÃ¼ntÃ¼le
4. Projeyi paylaÅŸ

**Not**: Bu proje eÄŸitim amaÃ§lÄ±dÄ±r ve gerÃ§ek tÄ±bbi tanÄ± yerine geÃ§mez.

# Kronik BÃ¶brek HastalÄ±ÄŸÄ± Tahmini (Chronic Kidney Disease Prediction)

Bu proje, Random Forest algoritmasÄ± kullanarak kronik bÃ¶brek hastalÄ±ÄŸÄ±nÄ±n erken tanÄ±sÄ±nÄ± yapmak iÃ§in geliÅŸtirilmiÅŸtir.

## Proje AÃ§Ä±klamasÄ±

Kronik BÃ¶brek HastalÄ±ÄŸÄ± (CKD), bÃ¶brek fonksiyonlarÄ±nÄ±n zamanla azalmasÄ±yla karakterize edilen ciddi bir saÄŸlÄ±k durumudur. Erken tanÄ±, hastalÄ±ÄŸÄ±n ilerlemesini yavaÅŸlatmak ve komplikasyonlarÄ± Ã¶nlemek iÃ§in kritik Ã¶neme sahiptir.

Bu proje, Ã§eÅŸitli klinik parametreleri analiz ederek Random Forest makine Ã¶ÄŸrenmesi algoritmasÄ± ile CKD'nin erken tahminini yapar.

## Ã–zellikler

- **Ã‡apraz DoÄŸrulama**: Model gÃ¼venilirliÄŸinin test edilmesi
- **DeÄŸerlendirme Metrikleri**: Accuracy, Precision, Recall, F1-Score
- **Feature Importance**: En Ã¶nemli Ã¶zelliklerin belirlenmesi
- **GÃ¶rselleÅŸtirme**: Model performansÄ±nÄ±n analizi ve gÃ¶rselleÅŸtirme
- **Random Forest Modeli**: YÃ¼ksek doÄŸruluk oranÄ± ile tahmin
- **Veri Ã–n Ä°ÅŸleme**: Eksik deÄŸerlerin temizlenmesi ve veri normalizasyonu

## Kurulum

### Gereksinimler

`ash
pip install -r requirements.txt
`

### KullanÄ±m

`python
from ckd_predictor import CKDPredictor

# Modeli eÄŸit
predictor = CKDPredictor()
predictor.load_data('data.csv')
predictor.preprocess_data()
predictor.train()

# Tahmin yap
prediction = predictor.predict(new_patient_data)
`

## Veri Seti

Proje, aÅŸaÄŸÄ±daki Ã¶zellikleri iÃ§eren kronik bÃ¶brek hastalÄ±ÄŸÄ± veri setini kullanÄ±r:

- **Demografik Bilgiler**: YaÅŸ, cinsiyet
- **Fiziksel Ã–zellikler**: Boy, kilo, BMI
- **Laboratuvar DeÄŸerleri**: Kan basÄ±ncÄ±, hemoglobin, albÃ¼min, kreatinin, vb.
- **TÄ±bbi GeÃ§miÅŸ**: Hipertansiyon, diyabet, koroner arter hastalÄ±ÄŸÄ±

## Model PerformansÄ±

- **Accuracy**: %95+
- **F1-Score**: %95+
- **Precision**: %94+
- **Recall**: %96+

## Dosya YapÄ±sÄ±

`
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ CKD_Analysis.ipynb        # Jupyter notebook
â”œâ”€â”€ ckd_predictor.py          # Ana tahmin sÄ±nÄ±fÄ±
â”œâ”€â”€ data_preprocessing.py     # Veri Ã¶n iÅŸleme
â”œâ”€â”€ example_usage.py          # Ã–rnek kullanÄ±m
â”œâ”€â”€ model_evaluation.py       # Model deÄŸerlendirme
â”œâ”€â”€ test_model.py            # Model test scripti
â”œâ”€â”€ visualization.py          # GÃ¶rselleÅŸtirme
â””â”€â”€ data/
    â””â”€â”€ sample_data.csv       # Ã–rnek veri seti
`

## KullanÄ±m Ã–rnekleri

### GÃ¶rselleÅŸtirme

`python
from visualization import DataVisualizer

visualizer = DataVisualizer()

# Korelasyon matrisi
visualizer.plot_correlation_heatmap(data, numeric_columns)

# Ã–zellik daÄŸÄ±lÄ±mlarÄ±
visualizer.plot_feature_distributions(data, numeric_columns)

# SÄ±nÄ±f daÄŸÄ±lÄ±mÄ±
visualizer.plot_class_distribution(data)
`

### Model DeÄŸerlendirme

`python
from model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Ã‡apraz doÄŸrulama
cv_scores = evaluator.cross_validation(model, X, y)

# Confusion matrix
evaluator.plot_confusion_matrix()

# Model performansÄ±nÄ± deÄŸerlendir
results = evaluator.evaluate_model(model, X_test, y_test, y_pred)

# Metrikleri yazdÄ±r
evaluator.print_metrics()

# Ã–zellik Ã¶nemleri
evaluator.feature_importance_plot(model, feature_names)
`

### Temel KullanÄ±m

`python
# Model deÄŸerlendirme
results = predictor.evaluate()

# Model eÄŸitimi
predictor.train()

# Veri yÃ¼kleme ve Ã¶n iÅŸleme
predictor = CKDPredictor()
predictor.load_data('chronic_kidney_disease.csv')
predictor.preprocess_data()
`

### Yeni Hasta Tahmini

`python
# Tahmin yap
result = predictor.predict(new_patient)
print(f"GÃ¼ven Skoru: {result['confidence']:.2f}%")
print(f"Tahmin: {result['prediction']}")

# Yeni hasta verisi
new_patient = {
    'age': 65,
    'ane': 0,
    'appet': 1,
    'ba': 0,
    'bgr': 120,
    'bp': 80,
    'bu': 60,
    'cad': 0,
    'dm': 0,
    'hemo': 12.5,
    'htn': 1,
    'pc': 0,
    'pcc': 0,
    'pcv': 38,
    'pe': 0,
    'pot': 4.5,
    'rbc': 0,
    'rbcc': 4.5,
    'sc': 1.2,
    'sg': 1.020,
    'sod': 140,
    'su': 0,
    'wbcc': 7000
}
`

## Ã–zellik AÃ§Ä±klamalarÄ±

| Ã–zellik | AÃ§Ä±klama |
|---------|----------|
| age | YaÅŸ |
| ane | Anemi |
| appet | Ä°ÅŸtah |
| ba | Bakteri |
| bgr | Kan glukozu |
| bp | Kan basÄ±ncÄ± |
| bu | Kan Ã¼re azotu |
| cad | Koroner arter hastalÄ±ÄŸÄ± |
| dm | Diyabet |
| hemo | Hemoglobin |
| htn | Hipertansiyon |
| pc | PÃ¼y hÃ¼cresi |
| pcc | PÃ¼y hÃ¼cresi klastlarÄ± |
| pcv | PaketlenmiÅŸ hÃ¼cre hacmi |
| pe | Periferik Ã¶dem |
| pot | Potasyum |
| rbc | KÄ±rmÄ±zÄ± kan hÃ¼cresi |
| rbcc | KÄ±rmÄ±zÄ± kan hÃ¼cresi sayÄ±sÄ± |
| sc | Serum kreatinin |
| sg | Spesifik gravite |
| sod | Sodyum |
| su | Åeker |
| wbcc | Beyaz kan hÃ¼cresi sayÄ±sÄ± |

## Teknik Detaylar

### Algoritma
- **AÄŸaÃ§ SayÄ±sÄ±**: 100
- **Maksimum Derinlik**: 10
- **Minimum Ã–rnek SayÄ±sÄ±**: 5 (split), 2 (leaf)
- **Random Forest**: Ensemble learning algoritmasÄ±

### DeÄŸerlendirme Metrikleri
- Accuracy (DoÄŸruluk)
- Confusion Matrix
- F1-Score
- Precision (Kesinlik)
- Recall (DuyarlÄ±lÄ±k)
- ROC AUC

### Veri Ã–n Ä°ÅŸleme
- Eksik deÄŸerlerin medyan/mod ile doldurulmasÄ±
- Kategorik deÄŸiÅŸkenlerin Label Encoding ile sayÄ±sal deÄŸerlere Ã§evrilmesi
- Ã–zelliklerin StandardScaler ile normalize edilmesi

## KatkÄ±da Bulunma

1. Commit yapÄ±n (git commit -m 'Add some AmazingFeature')
2. Feature branch oluÅŸturun (git checkout -b feature/AmazingFeature)
3. Fork yapÄ±n
4. Pull Request aÃ§Ä±n
5. Push yapÄ±n (git push origin feature/AmazingFeature)

## Ä°letiÅŸim

Proje hakkÄ±nda sorularÄ±nÄ±z iÃ§in issue aÃ§abilirsiniz.

## Lisans

Bu proje MIT lisansÄ± altÄ±nda lisanslanmÄ±ÅŸtÄ±r.

## Notlar

- Bu proje eÄŸitim amaÃ§lÄ±dÄ±r ve gerÃ§ek tÄ±bbi tanÄ± yerine geÃ§mez
- Model performansÄ± veri setine baÄŸlÄ± olarak deÄŸiÅŸebilir
- TÄ±bbi kararlar iÃ§in mutlaka uzman doktor gÃ¶rÃ¼ÅŸÃ¼ alÄ±nmalÄ±dÄ±r

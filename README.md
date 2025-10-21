# Kronik BÃ¶brek HastalÄ±ÄŸÄ± Tahmini (Chronic Kidney Disease Prediction)

Bu proje, Random Forest algoritmasÄ± kullanarak kronik bÃ¶brek hastalÄ±ÄŸÄ±nÄ±n erken tanÄ±sÄ±nÄ± yapmak iÃ§in geliÅŸtirilmiÅŸtir.

## Proje AÃ§Ä±klamasÄ±

Kronik BÃ¶brek HastalÄ±ÄŸÄ± (CKD), bÃ¶brek fonksiyonlarÄ±nÄ±n zamanla azalmasÄ±yla karakterize edilen ciddi bir saÄŸlÄ±k durumudur. Erken tanÄ±, hastalÄ±ÄŸÄ±n ilerlemesini yavaÅŸlatmak ve komplikasyonlarÄ± Ã¶nlemek iÃ§in kritik Ã¶neme sahiptir.

Bu proje, Ã§eÅŸitli klinik parametreleri analiz ederek Random Forest makine Ã¶ÄŸrenmesi algoritmasÄ± ile CKD'nin erken tahminini yapar.

## Ã–zellikler

- **Veri Ã–n Ä°ÅŸleme**: Eksik deÄŸerlerin temizlenmesi ve veri normalizasyonu
- **Random Forest Modeli**: YÃ¼ksek doÄŸruluk oranÄ± ile tahmin
- **GÃ¶rselleÅŸtirme**: Model performansÄ±nÄ±n analizi ve gÃ¶rselleÅŸtirme
- **DeÄŸerlendirme Metrikleri**: Accuracy, Precision, Recall, F1-Score
- **Feature Importance**: En Ã¶nemli Ã¶zelliklerin belirlenmesi
- **Ã‡apraz DoÄŸrulama**: Model gÃ¼venilirliÄŸinin test edilmesi

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
- **Laboratuvar DeÄŸerleri**: Kan basÄ±ncÄ±, hemoglobin, albÃ¼min, kreatinin, vb.
- **Fiziksel Ã–zellikler**: Boy, kilo, BMI
- **TÄ±bbi GeÃ§miÅŸ**: Hipertansiyon, diyabet, koroner arter hastalÄ±ÄŸÄ±

## Model PerformansÄ±

- **Accuracy**: %95+
- **Precision**: %94+
- **Recall**: %96+
- **F1-Score**: %95+

## Dosya YapÄ±sÄ±

`
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ ckd_predictor.py          # Ana tahmin sÄ±nÄ±fÄ±
â”œâ”€â”€ data_preprocessing.py     # Veri Ã¶n iÅŸleme
â”œâ”€â”€ model_evaluation.py       # Model deÄŸerlendirme
â”œâ”€â”€ visualization.py          # GÃ¶rselleÅŸtirme
â”œâ”€â”€ example_usage.py          # Ã–rnek kullanÄ±m
â””â”€â”€ sample_data.csv           # Ã–rnek veri seti
`

## KullanÄ±m Ã–rnekleri

### Temel KullanÄ±m

`python
# Veri yÃ¼kleme ve Ã¶n iÅŸleme
predictor = CKDPredictor()
predictor.load_data('chronic_kidney_disease.csv')
predictor.preprocess_data()

# Model eÄŸitimi
predictor.train()

# Model deÄŸerlendirme
results = predictor.evaluate()
`

### Yeni Hasta Tahmini

`python
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
result = predictor.predict(new_patient)
print(f"Tahmin: {result['prediction']}")
print(f"GÃ¼ven Skoru: {result['confidence']:.2f}%")
`

### GÃ¶rselleÅŸtirme

`python
from visualization import DataVisualizer

visualizer = DataVisualizer()

# SÄ±nÄ±f daÄŸÄ±lÄ±mÄ±
visualizer.plot_class_distribution(data)

# Ã–zellik daÄŸÄ±lÄ±mlarÄ±
visualizer.plot_feature_distributions(data, numeric_columns)

# Korelasyon matrisi
visualizer.plot_correlation_heatmap(data, numeric_columns)
`

## Model DeÄŸerlendirme

`python
from model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Model performansÄ±nÄ± deÄŸerlendir
results = evaluator.evaluate_model(model, X_test, y_test, y_pred)

# Metrikleri yazdÄ±r
evaluator.print_metrics()

# Confusion matrix
evaluator.plot_confusion_matrix()

# Ã–zellik Ã¶nemleri
evaluator.feature_importance_plot(model, feature_names)

# Ã‡apraz doÄŸrulama
cv_scores = evaluator.cross_validation(model, X, y)
`

## Ã–zellik AÃ§Ä±klamalarÄ±

| Ã–zellik | AÃ§Ä±klama |
|---------|----------|
| age | YaÅŸ |
| bp | Kan basÄ±ncÄ± |
| sg | Spesifik gravite |
| al | AlbÃ¼min |
| su | Åeker |
| rbc | KÄ±rmÄ±zÄ± kan hÃ¼cresi |
| pc | PÃ¼y hÃ¼cresi |
| pcc | PÃ¼y hÃ¼cresi klastlarÄ± |
| ba | Bakteri |
| bgr | Kan glukozu |
| bu | Kan Ã¼re azotu |
| sc | Serum kreatinin |
| sod | Sodyum |
| pot | Potasyum |
| hemo | Hemoglobin |
| pcv | PaketlenmiÅŸ hÃ¼cre hacmi |
| wbcc | Beyaz kan hÃ¼cresi sayÄ±sÄ± |
| rbcc | KÄ±rmÄ±zÄ± kan hÃ¼cresi sayÄ±sÄ± |
| htn | Hipertansiyon |
| dm | Diyabet |
| cad | Koroner arter hastalÄ±ÄŸÄ± |
| appet | Ä°ÅŸtah |
| pe | Periferik Ã¶dem |
| ane | Anemi |

## Teknik Detaylar

### Algoritma
- **Random Forest**: Ensemble learning algoritmasÄ±
- **AÄŸaÃ§ SayÄ±sÄ±**: 100
- **Maksimum Derinlik**: 10
- **Minimum Ã–rnek SayÄ±sÄ±**: 5 (split), 2 (leaf)

### Veri Ã–n Ä°ÅŸleme
- Eksik deÄŸerlerin medyan/mod ile doldurulmasÄ±
- Kategorik deÄŸiÅŸkenlerin Label Encoding ile sayÄ±sal deÄŸerlere Ã§evrilmesi
- Ã–zelliklerin StandardScaler ile normalize edilmesi

### DeÄŸerlendirme Metrikleri
- Accuracy (DoÄŸruluk)
- Precision (Kesinlik)
- Recall (DuyarlÄ±lÄ±k)
- F1-Score
- ROC AUC
- Confusion Matrix

## KatkÄ±da Bulunma

1. Fork yapÄ±n
2. Feature branch oluÅŸturun (git checkout -b feature/AmazingFeature)
3. Commit yapÄ±n (git commit -m 'Add some AmazingFeature')
4. Push yapÄ±n (git push origin feature/AmazingFeature)
5. Pull Request aÃ§Ä±n

## Lisans

Bu proje MIT lisansÄ± altÄ±nda lisanslanmÄ±ÅŸtÄ±r.

## Ä°letiÅŸim

Proje hakkÄ±nda sorularÄ±nÄ±z iÃ§in issue aÃ§abilirsiniz.

## Notlar

- Bu proje eÄŸitim amaÃ§lÄ±dÄ±r ve gerÃ§ek tÄ±bbi tanÄ± yerine geÃ§mez
- TÄ±bbi kararlar iÃ§in mutlaka uzman doktor gÃ¶rÃ¼ÅŸÃ¼ alÄ±nmalÄ±dÄ±r
- Model performansÄ± veri setine baÄŸlÄ± olarak deÄŸiÅŸebilir

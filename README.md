# 🐚 Abalone Tahmin Projesi

Bu proje, Abalone veri setini kullanarak deniz kabuklularının yaşını tahmin etmeyi amaçlamaktadır. Veri seti, farklı fiziksel özelliklerin yanı sıra, her bir abalone için halkaların sayısını içermektedir. Projede, XGBoost ve Random Forest gibi makine öğrenimi modelleri kullanılarak tahminler yapılmaktadır.

⚠️ Not
3D grafiklerim ve görselleştirmelerim maalesef gözükmüyor. Bu durum, bazı tarayıcı veya platform uyumsuzluklarından kaynaklanabilir.

## 🔗 Kaggle Veri Seti
[Abalone Dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

#### [Abalone Dataset - Hugging Face](https://huggingface.co/spaces/btulftma/abalone_dataset)

## 📊 Proje Aşamaları
1. **Veri Yükleme**:
   - Veri setleri `train.csv` ve `test.csv` dosyalarından yüklenir.
  
2. **Veri Keşfi ve Ön İşleme**:
   - İlk 5 satır ve temel istatistikler görüntülenir.
   - Eksik değerler kontrol edilir.
   - Cinsiyet bilgisi sayısal verilere dönüştürülür.
   - Hacim ve ağırlık oranları hesaplanır.

3. **Outlier Temizleme**:
   - Belirlenen özellikler için outlier'lar kırpılır.

4. **Modelleme**:
   - Özellikler ve hedef değişken hazırlanır.
   - Eğitim ve doğrulama setlerine bölünür.
   - XGBoost ve Random Forest modelleri oluşturulur ve eğitilir.
   - Her iki model için RMSE hesaplanır.

5. **Tahmin ve Sonuç**:
   - Test seti üzerinde tahminler yapılır ve sonuçlar `abalone_submission.csv` dosyasına kaydedilir.

6. **Görselleştirme**:
   - Özniteliklerin önem dereceleri ve gerçek-tahmin karşılaştırmaları görselleştirilir.
   - Korelasyon matrisi ve halka dağılımı grafikleri oluşturulur.

## 📈 Model Performansı
- **XGBoost RMSE**: `xgb_rmse` değişkeninde saklanır.
- **Random Forest RMSE**: `rf_rmse` değişkeninde saklanır.

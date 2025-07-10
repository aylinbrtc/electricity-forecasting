# ELECTRICITY FORECASTING PROJECT

## Proje Amacı

Bu proje, saatlik elektrik türetimi, tüketimi, fiyat verisi ve hava durumu bilgilerini kullanarak elektrik fiyatlarını tahmin etmeye yöneliktir. Proje iki temel aşamadan oluşur:

1. Saatlik bazda bir sonraki saatin fiyat tahmini
2. Günlük bazda bir sonraki günün ortalama fiyat tahmini

Veri üzerinde detaylı ön işleme, öznitelik mühendisliği, farklı model denemeleri ve son olarak ensemble yapı kurularak tahmin performansı maksimum seviyeye çıkarılmaya çalışılmıştır.

## Proje Yapısı

```
ELECTRICITY_FORECASTING_PROJECT
|— data
|   |— raw                # Ham veriler
|   |— processed          # Temizlenmiş ve birleştirilmiş veriler
|— notebooks
|   |— models            # Saatlik modeller
|   |— models_daily      # Günlük modeller
|   |— .ipynb files      # Notebooklar
```

## Kullanılan Veriler

* **energy\_dataset.csv**: Elektrik fiyatı, üretim ve tüketim verileri (saatlik)
* **weather\_features.csv**: Aynı zaman aralığında hava durumu verileri
* Temizlik ve birleştirme adımları sonunda: `energy_clean.csv`, `weather_clean.csv`, `merged_with_weather.csv`

## Notebook Açıklamaları

### 0\_data\_exploration.ipynb

* Verinin genel yapısı, eksik değerler, dağılım analizleri ve zamansal düzende trendler incelenmiştir.
* ACF/PACF, haftalık ve saatlik döngüler, fiyat-tüketim ilişkisi gibi yapılar ortaya konmuştur.
* Çıkarım: Lag temelli ve zaman tabanlı öznitelikler bilgi taşıyor.

### 1\_feature\_engineering.ipynb

* Saatlik tahmin için zaman, lag, rolling, frekans, hava durumu ve üretim bazlı öznitelikler üretildi.
* Özellikle `price_lag_1`, `load_lag_24` gibi özelliklerin etkili olduğu gözlendi.
* Korelasyon analizi ve LightGBM ile ön önem kontrolü yapıldı.

### 2\_model\_training.ipynb

* MLP, LightGBM ve Random Forest modelleri RandomizedSearchCV ile optimize edildi.
* TimeSeriesSplit kullanılarak zaman tabanlı doğrulama yapıldı.
* En iyi modeller kaydedildi, daha sonra ensemble modeli için kullanıldı.

### 3\_model\_testing.ipynb

* Tüm modeller test setinde değerlendirildi.
* Ağırlık optimizasyonu ile ensemble tahmini hesaplandı.
* SMAPE, MAE, RMSE gibi metrikler kullanıldı.
* Test setinde en iyi başarıyı ensemble model gösterdi.

### 1\_feature\_engineering\_daily.ipynb

* Günlük tahmin için hedef değişken: 24 saat sonrasının ortalama fiyatı.
* 6 gün geriye kadar lag ve rolling öznitelikler kullanıldı.
* Frekans tabanlı öznitelikler ve üretim tipine göre toplam enerji hesaplandı.
* LightGBM ile ön öznitelik seçimi yapıldı.

### 2\_model\_training\_daily.ipynb

* Günlük tahmin için MLP, RF ve LightGBM modelleri eğitildi.
* En iyi performansı LightGBM gösterdiği için ensemble ağırlığı %100 ona verildi.

### 3\_model\_testing\_daily.ipynb

* Günlük tahmin test setinde değerlendirildi.
* Ensemble modeli tek başına LightGBM çıktı.
* SMAPE: %3.69 gibi düşük hata oranı elde edildi.

## Modeller

* **MLP**: TensorFlow ile KerasRegressor, erken durdurma ile optimize edildi.
* **LightGBM**: En düşük hata oranını verdi, hem saatlik hem günlük tahminlerde başarılı.
* **Random Forest**: Ortalama performans, ensembleda katkısı düşük kaldı.
* **Ensemble**: Ağırlıklar optimize edilerek final tahmin hesaplandı. Günlük modelde ağırlıklar tek modele kaydı.

## Kayıtlanan Çıktılar

* `models/` ve `models_daily/` klasörlerinde modellerin .pkl ve .keras formatları
* `ensemble_weights.pkl` ve `important_features.txt` gibi destek dosyaları

## Nasıl Çalıştırılır?

1. `data/raw` klasörü altındaki ham verilerle başlanır.
2. `0_data_exploration.ipynb` ile veri incelenir ve yapılar anlaşılır.
3. `1_feature_engineering*.ipynb` dosyaları ile öznitelikler oluşturulur.
4. `2_model_training*.ipynb` dosyaları modelleri eğitir ve kaydeder.
5. `3_model_testing*.ipynb` dosyaları test verileriyle modelleri değerlendirir.

> Not: `models/` ve `models_daily/` klasörüne önceden kayıtlı modeller yüklenmişe şekilde notebooklar test edilmiştir.

## Modellerin İndirme Bağlantısı

Proje boyut sınırlarından dolayı modeller Google Drive üzerinden paylaşıldı:

- [Saatlik Modelleri İndir (Google Drive)](https://drive.google.com/drive/folders/1weQCBSra5TxtagRcx5BC24TNGQZS3R7R?usp=sharing)
- [Günlük Modelleri İndir (Google Drive)](https://drive.google.com/drive/folders/1_2WOjlDoAXM65KKqI2VOo39wiyq46iUM?usp=sharing)

İndirdiğiniz dosyaları projenin doğru dizinine eklediğinize emin olun.

## Sonuç

Bu proje, zaman serisi tabanlı elektrik fiyat tahmini konusunda detaylı bir öznitelik mühendisliği ve modelleme süreci sunmuştur. Ensemble yapılar ile tahmin kalitesi artırılmış, LightGBM modeli ise genel olarak en tutarlı sonuçları vermiştir. Yapılan çalışma veri sızıntısına karşı dikkatli yaklaşmış ve gerçekçi bir tahmin sistematiği ortaya koymuştur.

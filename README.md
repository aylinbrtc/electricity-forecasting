# ELEKTRİK FİYATI TAHMİN PROJESİ

## Proje Amacı

Bu proje, saatlik elektrik üretimi, tüketimi, fiyat verileri ve hava durumu bilgilerini kullanarak elektrik fiyatlarını tahmin etmeye odaklanmıştır. Proje, iki farklı metodolojik yaklaşım üzerinden iki temel hedefi gerçekleştirmeyi amaçlar:

1.  **Saatlik Tahmin**: Bir sonraki saatin anlık elektrik fiyatını tahmin etme.
2.  **Günlük Tahmin**: Bir sonraki 24 saatin ortalama elektrik fiyatını tahmin etme.

Proje kapsamında, zaman serisi verileri üzerinde detaylı ön işleme, kapsamlı öznitelik mühendisliği ve farklı model mimarileri denenmiştir. Proje, klasik makine öğrenmesi modelleriyle oluşturulmuş bir **ensemble** yaklaşımını ve derin öğrenme tabanlı (RNN) daha gelişmiş ikinci bir yaklaşımı içermektedir.

## Proje Yapısı

Proje, iki ana metodolojiyi barındıracak şekilde yeniden organize edilmiştir.

```
ELECTRICITY_FORECASTING_PROJECT
|
|— data
| |— raw/ # Ham veriler (energy_dataset.csv, weather_features.csv)
| |— processed/ # İşlenmiş, temizlenmiş ve modellere hazır veriler
|
|— models/ # İkinci Yöntem (RNN) ile eğitilmiş nihai modeller (.keras, .joblib)
|
|— notebooks
| |— first_method/ # Yöntem 1: Klasik ML ve Ensemble Modelleri Notebook'ları
| | |— models/ # Yöntem 1 için saatlik modeller ve çıktılar
| | |— models_daily/ # Yöntem 1 için günlük modeller ve çıktılar
| |
| |— second_method/ # Yöntem 2: Derin Öğrenme (RNN) Modelleri Notebook'ları
| |— 1_feature_enginering_master.ipynb
| |— 2_model_training_hourly.ipynb
| |— 2_model_training_daily.ipynb
| |— ... (test notebook'ları)
|
|— requirements.txt # Gerekli kütüphaneler
|— README.md # Bu dosya
```

## Kullanılan Veriler

*   **energy\_dataset.csv**: Saatlik elektrik fiyatı, üretim ve tüketim verileri.
*   **weather\_features.csv**: Eş zamanlı hava durumu verileri.
*   **İşlenmiş Veriler (`data/processed/`)**:
    *   `merged_with_weather.csv`: Temizlenmiş ve birleştirilmiş ana veri seti.
    *   `hourly_model_data.npz`: İkinci yöntem için hazırlanmış saatlik model sekansları.
    *   `daily_model_data.npz`: İkinci yöntem için hazırlanmış günlük model sekansları.
    *   `scaler_*.joblib`: Verileri ölçeklendirmek için kullanılan scaler nesneleri.

---

## Model ve Veri İndirme Bağlantıları

Proje adımlarını atlayarak doğrudan test veya analiz yapmak isterseniz, önceden eğitilmiş modelleri ve işlenmiş verileri aşağıdaki Google Drive bağlantılarından indirebilirsiniz.

*   **Metodoloji 1 - Klasik Modeller:**
    *   [Saatlik Modelleri İndir (Yöntem 1)](https://drive.google.com/drive/folders/1weQCBSra5TxtagRcx5BC24TNGQZS3R7R?usp=sharing)
    *   [Günlük Modelleri İndir (Yöntem 1)](https://drive.google.com/drive/folders/1_2WOjlDoAXM65KKqI2VOo39wiyq46iUM?usp=sharing)

*   **Metodoloji 2 - Derin Öğrenme Modelleri:**
    *   [Nihai RNN Modellerini İndir (`models/` klasörü)](https://drive.google.com/drive/folders/1eYvCQe2UOpDiJmynQxSMHiXBObG5ozn_?usp=drive_link)

*   **İşlenmiş Veri Setleri (`.npz`):**
    *   [İşlenmiş Saatlik ve Günlük Verileri İndir (`data/processed/`)](https://drive.google.com/drive/folders/1_0FyLgLsbPYNBFzm5DBfoSul42t_a6GJ?usp=drive_link)

---

## Metodoloji 1: Klasik ML ve Ensemble Modelleri

Bu yaklaşım, projenin ilk aşamasını oluşturur ve `notebooks/first_method` klasöründe yer alır. Temel olarak lag/rolling gibi geleneksel özniteliklere dayalı ve farklı modellerin birleşiminden oluşan bir yapı kullanır.

### Notebook Açıklamaları (`first_method/`)

*   **0\_data\_exploration.ipynb**: Veri keşfi, döngüsellik ve korelasyon analizleri.
*   **1\_feature\_engineering(\_daily).ipynb**: Saatlik ve günlük tahminler için lag, zaman ve hareketli ortalama tabanlı özniteliklerin üretilmesi.
*   **2\_model\_training(\_daily).ipynb**: `MLP`, `LightGBM` ve `Random Forest` modellerinin `RandomizedSearchCV` ile optimizasyonu ve `TimeSeriesSplit` ile doğrulanması.
*   **3\_model\_testing(\_daily).ipynb**: Modellerin test verisi üzerinde değerlendirilmesi ve ağırlıkları optimize edilmiş bir **ensemble** modelinin oluşturulması.

### Modeller ve Çıktılar

*   Bu yöntemde eğitilen tüm modeller (`.pkl`, `.keras`), scaler'lar ve diğer çıktılar kendi klasörleri olan `notebooks/first_method/models/` ve `notebooks/first_method/models_daily/` altında saklanır.
*   Bu modellere yukarıdaki indirme bölümünden ulaşabilirsiniz.
*   Ensemble yapı, özellikle saatlik tahminlerde tekil modellere kıyasla daha iyi performans göstermiştir.

---

## Metodoloji 2: Derin Öğrenme (Conv1D + GRU)

Bu yaklaşım, zaman serisi verilerindeki karmaşık ve doğrusal olmayan desenleri yakalamak için derin öğrenme modellerini kullanır. Tüm notebook'lar `notebooks/second_method` altında, nihai modeller ise ana `models/` dizininde yer alır.

### Notebook Açıklamaları (`second_method/`)

*   **1\_feature\_enginering\_master.ipynb**: Her iki (saatlik ve günlük) model için veri hazırlama sürecini merkezileştirir. Fourier ve Dalgacık dönüşümleri gibi sinyal işleme teknikleriyle gelişmiş öznitelikler türetir. Çıktı olarak, RNN modellerine uygun, ölçeklendirilmiş ve sekanslanmış `hourly_model_data.npz` ve `daily_model_data.npz` dosyalarını üretir.
*   **2\_model\_training\_hourly.ipynb**: Geçmiş 24 saatlik veriyi kullanarak bir sonraki saatin fiyatını tahmin eden **GRU tabanlı** bir RNN modeli eğitir. `KerasTuner` ile hiperparametre optimizasyonu yapar.
*   **2\_model\_training\_daily.ipynb**: Bir sonraki 24 saatin ortalama fiyatını tahmin eden, daha gelişmiş bir model mimarisi kullanır. Bu mimari:
    *   **Conv1D Katmanı**: Zaman serisindeki gürültüyü filtreler ve kısa vadeli desenleri bir "desen avcısı" gibi yakalar.
    *   **İstiflenmiş GRU (Stacked GRU)**: Birbiri ardına gelen GRU katmanları ile desenler arası hiyerarşik ilişkileri öğrenir, bu da gürültülü günlük tahminlerde performansı önemli ölçüde artırır.
*   **3\_model\_testing(\_hourly/\_daily).ipynb**: Eğitilen nihai modellerin test verisi üzerindeki performansını (MAE, RMSE, R²) metrikleriyle değerlendirir.

### Modeller ve Çıktılar

*   Bu yöntemin en iyi performans gösteren nihai modelleri ana `models/` klasörüne kaydedilmiştir. Bu modellere yukarıdaki indirme bölümünden ulaşabilirsiniz.
    *   `best_tuned_hourly_model.keras`: Optimize edilmiş saatlik tahmin modeli.
    *   `best_tuned_daily_model.keras`: Optimize edilmiş günlük tahmin modeli (Conv1D + Stacked GRU).
*   Modellerin kullandığı ölçekleyiciler ve `.npz` formatındaki işlenmiş veriler ise `data/processed/` klasöründe saklanır. Bu dosyalara da yukarıdaki indirme bağlantılarından erişilebilir.

## Nasıl Çalıştırılır?

Projeyi çalıştırmak için iki metodolojiden birini seçebilir veya önceden eğitilmiş çıktıları kullanabilirsiniz:

1.  **Metodoloji 1'i Çalıştırmak İçin:**
    *   `notebooks/first_method/` dizinine gidin.
    *   Notebook'ları `0_`, `1_`, `2_`, `3_` sırasında çalıştırın. Alternatif olarak, ilgili modelleri yukarıdaki linkten indirip test (`3_*`) notebook'larını doğrudan çalıştırın.

2.  **Metodoloji 2'yi Çalıştırmak İçin:**
    *   `notebooks/second_method/` dizinine gidin.
    *   **Seçenek A (Sıfırdan):** Önce `1_feature_enginering_master.ipynb` dosyasını çalıştırarak verileri oluşturun. Sonra `2_model_training_*.ipynb` dosyalarıyla modelleri eğitin.
    *   **Seçenek B (Hızlı Test):** Yukarıdaki linklerden **Nihai RNN Modellerini** `models/` klasörüne ve **İşlenmiş Veri Setlerini** `data/processed/` klasörüne indirin. Ardından `3_model_testing_*.ipynb` dosyalarını çalıştırarak sonuçları doğrudan değerlendirin.

## Sonuç

Bu proje, elektrik fiyatı tahmini problemini iki farklı ve güçlü yaklaşımla ele almıştır.
*   **İlk Yöntem**, LightGBM gibi modellere dayalı **ensemble** yapısının ne kadar etkili ve hızlı bir çözüm olabileceğini göstermiştir.
*   **İkinci Yöntem** ise, Conv1D ve İstiflenmiş GRU gibi derin öğrenme mimarilerinin, özellikle gürültülü ve karmaşık zaman serisi verilerinde desenleri ne kadar derinden öğrenebildiğini ortaya koymuştur.

# License Plate Detection System

Bu proje, araç plakalarını tespit etmek ve okumak için YOLOv8 ve EasyOCR modellerini kullanarak gerçek zamanlı video işleme sağlayan bir sistem sunar. Streamlit uygulaması üzerinden görüntü ve video yükleme, canlı kamera akışı ile plaka tanıma işlemleri yapılabilir.

## Özellikler

- **Görüntü Yükleme**: Yüklenen bir araç görüntüsünde plaka tespiti ve okuma.
- **Fotoğraf Çekme**: Web kamerası ile anlık fotoğraf çekme ve plaka tespiti.
- **Canlı Video Akışı**: Web kamerasından gerçek zamanlı video akışında plaka tespiti.
- **Video Yükleme**: Yüklenen video dosyalarını işleme ve plaka tespiti.

## Kullanılan Teknolojiler

- **YOLOv8**: Araç ve plaka tespiti için kullanılan nesne algılama modeli.
- **EasyOCR**: Plaka üzerindeki yazıları okumak için kullanılan optik karakter tanıma (OCR) kütüphanesi.
- **OpenCV**: Görüntü işleme işlemleri için kullanılan kütüphane.
- **Streamlit**: Kullanıcı arayüzü ve uygulama geliştirme framework'ü.

## Kurulum

1. Python 3.8+ ve gerekli kütüphaneler kurulu olmalıdır. Gerekli kütüphaneleri kurmak için:

    ```bash
    pip install -r requirements.txt
    ```

2. `models` klasörüne YOLOv8 ve plaka tespit modeli dosyalarını yerleştirin:
   - `yolov8n.pt`: YOLOv8 model dosyası.
   - `license_plate_detector.pt`: Plaka tespit modeli dosyası.

3. Uygulamayı başlatmak için:

    ```bash
    streamlit run app.py
    ```

## Kullanım

Uygulama başlatıldığında, aşağıdaki dört ana seçenek bulunur:

1. **Upload an Image**: Bilgisayarınızdan bir araç resmi yükleyin ve plaka tespiti yapın.
2. **Take a Photo**: Web kameranızı kullanarak anlık bir fotoğraf çekin ve plaka tespiti yapın.
3. **Live Detection**: Web kameranızdan gerçek zamanlı video akışı ile plaka tespiti yapın.
4. **Upload a Video**: Bilgisayarınızdan bir video yükleyin ve her karede plaka tespiti yapın.

## Notlar

- **Görselleştirme**: Plakaların ve araçların etrafında bulunan sınır kutuları ile birlikte tespit sonuçları gösterilir.
- **Geçici Dosyalar**: Geçici video dosyaları, Streamlit uygulamasının çalışması için kullanılmaktadır.

## Katkıda Bulunanlar

- Osman Temel (Geliştirici)


## İletişim

- [GitHub Profilim](https://github.com/osmantemel/)

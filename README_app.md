# Image Classification App

Aplikasi Streamlit untuk klasifikasi gambar menggunakan model deep learning yang telah dilatih sebelumnya.

## 📋 Deskripsi Proyek

Aplikasi ini menggunakan model `model_best` dari Aqsha untuk melakukan klasifikasi gambar. Pengguna dapat mengupload gambar dan mendapatkan prediksi kategori beserta tingkat kepercayaan (confidence score).

### ✨ Fitur Utama

- **Upload Gambar**: Mendukung format PNG, JPG, JPEG, BMP, TIFF
- **Preprocessing Otomatis**: Resize dan normalisasi gambar sesuai spesifikasi model
- **Prediksi Real-time**: Menampilkan hasil klasifikasi dengan confidence score
- **UI yang User-friendly**: Interface yang mudah digunakan dengan visualisasi yang jelas
- **Top 5 Predictions**: Menampilkan 5 prediksi teratas dengan confidence score

## 🚀 Cara Menjalankan Aplikasi

### Prerequisites

- Python 3.8 atau lebih tinggi
- Model `model_best`

### Instalasi

1. **Clone atau download proyek ini**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pastikan model tersedia:**
   - Letakkan file `model_best` (atau `model_best.h5`, `model_best.keras`) di direktori proyek
   - Aplikasi akan mencari model di lokasi berikut:
     - `model_best`
     - `model_best.h5`
     - `model_best.keras`
     - `models/model_best`
     - `models/model_best.h5`
     - `models/model_best.keras`

4. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

5. **Buka browser:**
   - Aplikasi akan otomatis terbuka di `http://localhost:8501`
   - Jika tidak otomatis terbuka, buka browser dan akses URL tersebut

## 📁 Struktur File

```
streamlit-binus/
├── app.py              # File aplikasi utama
├── requirements.txt     # Dependencies Python
├── README_app.md       # Dokumentasi ini
└── model_best          # Model yang telah dilatih (harus ada)
```

## 🔧 Konfigurasi Model

### Mengubah Class Labels

Jika model Anda memiliki class labels yang berbeda, edit fungsi `get_class_labels()` di `app.py`:

```python
def get_class_labels():
    return [
        "Label Kelas 1",
        "Label Kelas 2", 
        "Label Kelas 3",
        # ... tambahkan sesuai dengan model Anda
    ]
```

### Mengubah Ukuran Input

Jika model Anda membutuhkan ukuran input yang berbeda, edit parameter `target_size` di fungsi `preprocess_image()`:

```python
def preprocess_image(image, target_size=(224, 224)):  # Ubah ukuran sesuai kebutuhan
```

## 📊 Cara Menggunakan

1. **Upload Gambar:**
   - Klik "Browse files" atau drag & drop gambar
   - Format yang didukung: PNG, JPG, JPEG, BMP, TIFF

2. **Lihat Hasil:**
   - Gambar asli dan yang sudah diproses akan ditampilkan
   - Prediksi utama dengan confidence score
   - Top 5 prediksi dengan confidence score masing-masing

3. **Interpretasi Hasil:**
   - **Predicted Class**: Kelas yang diprediksi oleh model
   - **Confidence**: Tingkat kepercayaan (0-1, semakin tinggi semakin yakin)
   - **Confidence Level**: Visualisasi bar progress

## 🛠️ Troubleshooting

### Model Tidak Ditemukan
```
❌ Model not found. Please ensure model_best is in the project directory.
```
**Solusi:** Pastikan file model ada di direktori proyek dengan nama yang benar.

### Error Loading Model
```
❌ Error loading model: [error message]
```
**Solusi:** 
- Periksa format model (h5, keras, atau saved model)
- Pastikan model kompatibel dengan versi TensorFlow yang digunakan
- Periksa struktur model dan dependencies

### Error Preprocessing Image
```
❌ Error preprocessing image: [error message]
```
**Solusi:**
- Pastikan gambar dalam format yang didukung
- Periksa ukuran gambar (terlalu besar bisa menyebabkan masalah)
- Pastikan gambar tidak corrupt

## 📝 Dependencies

- **streamlit**: Framework web app
- **tensorflow**: Deep learning framework
- **numpy**: Komputasi numerik
- **opencv-python**: Computer vision library
- **Pillow**: Image processing
- **pathlib2**: Path utilities

## 🔄 Update dan Maintenance

### Menambah Class Labels Baru
1. Edit fungsi `get_class_labels()` di `app.py`
2. Pastikan jumlah class sesuai dengan output model

### Mengubah Preprocessing
1. Edit fungsi `preprocess_image()` di `app.py`
2. Sesuaikan dengan kebutuhan model Anda

### Menambah Fitur Baru
1. Edit file `app.py` sesuai kebutuhan
2. Update `requirements.txt` jika ada dependency baru
3. Update dokumentasi ini

## 📞 Support

Jika mengalami masalah atau butuh bantuan:
1. Periksa bagian Troubleshooting di atas
2. Pastikan semua dependencies terinstall dengan benar
3. Periksa kompatibilitas model dengan TensorFlow version

---

**Happy Classifying! 🎉**

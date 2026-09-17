# Panduan Penempatan Dataset Patung Bali (8 Kelas)

Direktori ini disiapkan untuk menyimpan data akuisisi lapangan sesuai dengan SOP Pengambilan Data Lapangan CSBSI.

## Struktur 8 Kelas Tipologi Patung

Terdapat 8 kelas tipologi patung Bali, masing-masing memuat 2 model patung:

```text
datasets/
├── Dewa/
│   ├── model_01/
│   └── model_02/
├── Dewi/
│   ├── model_01/
│   └── model_02/
├── Mitologi/
│   ├── model_01/
│   └── model_02/
├── Penabuh/
│   ├── model_01/
│   └── model_02/
├── Pengapit/
│   ├── model_01/
│   └── model_02/
├── Punakawan/
│   ├── model_01/
│   └── model_02/
├── Raksasa/
│   ├── model_01/
│   └── model_02/
└── Wanara/
    ├── model_01/
    └── model_02/
```

## Struktur Subfolder di Dalam Setiap Model Patung

Untuk setiap model patung (misal: `datasets/Dewa/model_01/`), letakkan file sesuai perannya:

1. **`images/` (Data Utama - Wajib)**:
   - Letakkan seluruh foto multi-view orbit O1, O2, O3, dan foto detail close-up (format `.jpg`, `.png`, atau `.JPG`).
   - Sesuai SOP Bab 4.4, jumlah foto yang disarankan adalah 150–300 foto.
   - Pastikan foto papan identitas sesi ditaruh di folder log terpisah, bukan di dalam `images/`.

2. **`raw_video/` (Data Cadangan - Opsional)**:
   - Jika Anda merekam video orbit 4K sebagai cadangan, letakkan file videonya (misal `DJI_0216.MP4`) di sini.
   - Script `pipeline_tools/01_extract_frames.py` dapat mengekstrak video ini menjadi citra ke dalam folder `images/` secara otomatis jika foto langsung kurang/gagal.

3. **`ground_truth/` (Acuan Geometri LiDAR - Opsional/Evaluasi)**:
   - Letakkan file point cloud hasil pemindaian 3D LiDAR (format `.ply`) beserta informasi skala di sini.
   - File ini digunakan oleh script `pipeline_tools/05_evaluate_comparison.py` untuk menghitung metrik akurasi geometri (Chamfer Distance).

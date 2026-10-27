# Input dataset penelitian

Tersedia **18 model**: kelas `Dewa`, `Dewi`, `Mitologi`, `Penabuh`, `Pengapit`, `Punakawan`, `Raksasa`, dan `Wanara` masing-masing memiliki `model_01` dan `model_02`. Kelompok `TanksAndTemples` memiliki `tanks_and_temples_1` dan `tanks_and_temples_2`.

Buat seluruh folder dengan `python pipeline_tools/init_dataset_folders.py` dari root proyek. Perintah ini mempertahankan data yang sudah ada.

```text
datasets/<kelas>/<model>/
  input/          # Foto asli JPG/JPEG/PNG; nama unik, tanpa subfolder
  raw_video/      # Video opsional untuk ekstraksi frame ke input/
  ground_truth/   # lidar.ply opsional, dalam meter dan area objek yang sama
  images/         # Dihasilkan COLMAP: foto undistorsi
  distorted/      # Dihasilkan COLMAP: database/model sementara
  sparse/0/       # Dihasilkan COLMAP: cameras, images, points3D
```

Contoh: `datasets/Dewa/model_01/input/IMG_0001.JPG` dan `datasets/TanksAndTemples/tanks_and_temples_1/input/000001.jpg`.

Foto akuisisi lama pada `images/` masih didukung jika `input/` kosong. Jika tersedia model COLMAP siap pakai, isi `images/` serta tiga file model pada `sparse/0/`; pipeline memeriksa kelengkapan berkas sebelum melewati COLMAP. Simpan foto identitas sesi dan catatan lapangan di luar folder citra rekonstruksi.

Lihat [RUN_SERVER.md](../RUN_SERVER.md) untuk antrean dashboard, output, dan protokol evaluasi.

# Menjalankan eksperimen 18 model di server WSL Ubuntu 22.04

Dataset penelitian terdiri dari 16 togog Bali (8 kelas, masing-masing 2 model) dan 2 slot benchmark Tanks and Temples. Semua nama dan routing dataset memakai satu katalog di `pipeline_tools/dataset_registry.py`.

## 1. Siapkan environment di server

Ikuti [SETUP_WINDOWS11.md](SETUP_WINDOWS11.md) untuk driver Windows, WSL 2, compiler, COLMAP, Conda, dan CUDA Toolkit 11.8. Gunakan salinan workspace ini beserta perubahan lokalnya, bukan clone SuGaR upstream yang belum memuat pipeline skripsi. Simpan proyek dan dataset di filesystem Linux, misalnya `/home/anggara/SuGaR`.

```bash
cd ~/SuGaR
conda activate sugar
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
python pipeline_tools/check_environment.py --gpu 0
```

Pemeriksaan environment memvalidasi Python 3.9, PyTorch 2.0.1/CUDA 11.8, PyTorch3D 0.7.4, compiler, COLMAP, paket Python, serta menjalankan kernel kecil simple-knn, PyTorch3D KNN, dan Gaussian rasterizer. Hasilnya disimpan di `outputs/logs/environment_check.json`. Nama GPU dan kapasitas VRAM yang dilaporkan server menjadi acuan konfigurasi. Belum ada hasil training server yang disertakan di repository ini.

Untuk instalasi otomatis dari Ubuntu, tersedia `python install.py`; installer berhenti jika salah satu langkah gagal. Nvdiffrast dilewati secara default. Jika dibutuhkan, jalankan `python install.py --with_nvdiffrast` untuk versi v0.3.3. Jangan menaikkan PyTorch atau menginstal Nvdiffrast terbaru tanpa menguji ulang semua extension.

## 2. Buat folder dan letakkan data

```bash
bash setup_workspace.sh
# Alternatif:
python pipeline_tools/init_dataset_folders.py
```

Perintah aman dijalankan ulang: membuat direktori yang belum ada dan mempertahankan foto/model yang sudah tersimpan.

```text
datasets/
  Dewa/       model_01/ dan model_02/
  Dewi/       model_01/ dan model_02/
  Mitologi/   model_01/ dan model_02/
  Penabuh/    model_01/ dan model_02/
  Pengapit/   model_01/ dan model_02/
  Punakawan/  model_01/ dan model_02/
  Raksasa/    model_01/ dan model_02/
  Wanara/     model_01/ dan model_02/
  TanksAndTemples/
    tanks_and_temples_1/
    tanks_and_temples_2/
```

Isi setiap model dengan struktur berikut:

```text
datasets/<kelas>/<model>/
  input/          # INPUT: foto asli .jpg/.jpeg/.png, nama unik, satu level folder
  raw_video/      # INPUT opsional: video pengganti foto
  ground_truth/   # INPUT opsional: lidar.ply dalam meter, objek sudah dipotong
  images/         # DATA OLAHAN: foto undistorsi, dihasilkan COLMAP
  distorted/      # DATA OLAHAN: database dan model SfM
  sparse/0/       # DATA OLAHAN: cameras, images, points3D
```

Contoh foto: `datasets/Dewa/model_01/input/IMG_0001.JPG`. Foto pada `images/` dari struktur lama masih didukung jika `input/` kosong. File `.gitkeep` tidak dihitung sebagai foto atau checkpoint.

Untuk dua sampel Tanks and Temples, letakkan foto dari scene pilihan pada masing-masing `input/`. Bila sudah memiliki hasil COLMAP terundistorsi, masukkan `images/` dan tiga file model pada `sparse/0/`; proses dapat melewati COLMAP. Nama kedua slot mengikuti permintaan penelitian; catat nama scene resmi, sumber unduhan, dan unit ground truth dalam catatan eksperimen. Evaluasi lokal ini memakai split kamera yang sama dengan dataset Bali, bukan protokol leaderboard resmi Tanks and Temples.

## 3. Jalankan dashboard dan antrekan semua model

```bash
conda activate sugar
cd ~/SuGaR
bash start_dashboard.sh 8080
```

Buka `http://localhost:8080` dari browser Windows pada server yang sama. Untuk mengakses server melalui SSH dari komputer lain, gunakan terminal komputer klien:

```bash
ssh -L 8080:127.0.0.1:8080 anggara@alamat-server
```

Lalu buka `http://localhost:8080` di komputer klien. Alamat tunnel harus dapat mencapai layanan localhost WSL pada server. Dashboard secara default terikat ke `127.0.0.1`; konfigurasi jaringan SSH/WSL mengikuti server Anda.

Di dashboard:

1. Isi **GPU ID** sesuai indeks di `nvidia-smi`. Awali dengan refinement **short** dan target **250000 segitiga** untuk pilot. Iterasi refinement short/medium/long = 2000/7000/15000.
2. Isi foto setiap model, kemudian klik **Antrekan Semua 18 Model**. Dataset kosong dilaporkan sebagai dilewati; model yang memiliki data masuk FIFO sesuai urutan katalog.
3. Satu pekerjaan menjalankan tahap secara berurutan: ekstraksi video bila foto belum tersedia → COLMAP → 3DGS 30000 iterasi (menyimpan 7000 dan 30000) → SuGaR → evaluasi visual 3DGS → evaluasi visual SuGaR.
4. Anda bisa menambahkan pekerjaan baru ketika pekerjaan lain berjalan. Duplikat pekerjaan yang masih menunggu/berjalan tidak ditambahkan lagi. Evaluasi visual 3DGS dan SuGaR menjadi dua pekerjaan berbeda saat dijalankan lewat tombol tahap.
5. **Jeda Antrean** membiarkan pekerjaan aktif selesai lalu menahan pekerjaan berikutnya. **Stop** menghentikan seluruh pohon proses pekerjaan aktif dan menjeda antrean. Pekerjaan aktif ditandai dibatalkan; klik **Lanjutkan Antrean** untuk model berikutnya. Pekerjaan gagal dapat ditambahkan ulang.
6. Antrean disimpan di `outputs/queue/state.json`; log setiap pekerjaan di `outputs/queue/logs/<id>.log`. Klik **Log** pada baris pekerjaan untuk membaca hasil/errornya, atau **Live** untuk kembali ke proses aktif. Sesudah server dashboard dimulai ulang, antrean dijeda. Pekerjaan yang terputus ditandai `INTERRUPTED`, bukan sukses. Pastikan tidak ada proses training lama tersisa pada GPU sebelum melanjutkan.

Hanya satu instance dashboard per workspace yang dapat berjalan. Lock GPU juga dipakai oleh wrapper training/evaluasi CLI, sehingga wrapper dari workspace yang sama tidak menjalankan training bersamaan pada GPU yang sama. Program lain, salinan workspace lain, dan pemanggilan langsung kode upstream di luar wrapper tidak mengikuti lock ini.

Citra latih disimpan di RAM dan dipindahkan per view ke GPU. Setiap tahap utama berjalan sebagai subprocess terpisah agar alokasi GPU dilepas setelah keluar; cache juga dibersihkan antartahap SuGaR. Antrean mencegah tumpang tindih pekerjaan, tetapi tidak menjamin satu scene pasti muat di VRAM. Bila satu model OOM, catat kegagalannya, sesuaikan ukuran data/target mesh secara konsisten untuk eksperimen, lalu antrekan ulang. Pastikan kapasitas RAM WSL dan ruang disk mencukupi dataset, checkpoint, dan seluruh render evaluasi.

## 4. Lokasi hasil

```text
outputs/
  3dgs/<kelas>/<model>/
    cameras.json
    cfg_args
    chkpnt7000.pth
    chkpnt30000.pth
    point_cloud/iteration_7000/point_cloud.ply
    point_cloud/iteration_30000/point_cloud.ply
    evaluation/{metrics.json,renders/,ground_truth/}
  sugar/<kelas>/<model>/
    coarse/
    mesh/
    refined/
    textured_mesh/                 # OBJ + MTL + tekstur PNG
    pipeline_result.json           # path aktual checkpoint dan mesh
    evaluation/{metrics.json,renders/,ground_truth/}
  logs/
    environment_check.json
    training_logs/                 # JSON/log tiap tahap
    evaluation_reports/            # summary_comparison.csv dan .md
  queue/{state.json,logs/}
```

`pipeline_result.json` hanya ditulis sesudah pipeline SuGaR berhasil. Dashboard tidak menganggap folder kosong sebagai hasil selesai. Tampilan preview menggunakan render citra uji, bukan gambar atlas tekstur OBJ.

**Lewati tahap yang selesai** memeriksa file hasil dan konfigurasi refinement/target mesh yang diminta. Evaluasi memakai signature checkpoint untuk menghindari hasil lama setelah checkpoint diganti. Ini melanjutkan antartahap, bukan melanjutkan iterasi optimizer yang terputus. Bila dataset diubah, nonaktifkan opsi lewati dan gunakan folder dataset eksperimen baru agar model dan evaluasi lama tidak tercampur.

## 5. Evaluasi dan batas interpretasi

Pipeline lengkap menghitung PSNR, SSIM, dan LPIPS VGG pada setiap kamera kedelapan yang dikeluarkan dari pelatihan. Kedua metode memakai pembagian kamera dan resolusi evaluasi yang sama. Input LPIPS dinormalisasi ke [-1, 1] sesuai [dokumentasi resmi LPIPS](https://github.com/richzhang/PerceptualSimilarity#1-learned-perceptual-image-patch-similarity-lpips-metric). SuGaR dievaluasi sebagai representasi hybrid mesh + Gaussians; skor ini bukan skor rendering OBJ bertekstur konvensional. LPIPS dapat mengunduh bobot pretrained pada pemakaian pertama; siapkan jaringan/cache bobot di server sebelum eksperimen panjang.

Evaluasi geometri LiDAR dijalankan terpisah setelah memilih metode dan mengisi skala meter per unit model. Skala harus diukur; pipeline tidak mengasumsikan COLMAP menghasilkan meter. Ground truth harus memakai meter dan area objek yang sama. ICP memerlukan orientasi awal cukup dekat dan sekarang menolak overlap yang terlalu rendah. Selalu periksa hasil alignment sebelum memakai Chamfer di skripsi. Pusat Gaussian 3DGS dan sampel permukaan mesh SuGaR adalah representasi geometri berbeda dan perlu dijelaskan dalam metodologi.

CSV berisi histori sesi training dan evaluasi. Dashboard menggabungkan nilai terakhir yang sukses per kelas/model/metode dan menampilkan waktu/VRAM training terpisah dari sesi evaluasi. VRAM yang tidak dapat diukur pada WSL ditampilkan `-`, bukan dianggap 0 GB. Pilih parameter final yang sama dan catat versi environment untuk seluruh 18 model sebelum mengambil hasil akhir skripsi.

## 6. Menjalankan tanpa dashboard

```bash
# Seluruh 18 model, tetap satu per satu:
python -u pipeline_tools/batch_run_all.py --gpu 0

# Satu togog:
python -u pipeline_tools/batch_run_all.py -c Dewa -m model_01 --gpu 0

# Kedua benchmark:
python -u pipeline_tools/batch_run_all.py -c TanksAndTemples --gpu 0

# Satu benchmark:
python -u pipeline_tools/batch_run_all.py -c TanksAndTemples -m tanks_and_temples_1 --gpu 0
```

Gunakan dashboard sebagai pengendali utama saat menjalankan antrean panjang. CLI mengembalikan exit code bukan nol jika ada model gagal, tetap mencetak ringkasan kegagalan, dan tidak mengimpor modul training GPU ke proses supervisor.

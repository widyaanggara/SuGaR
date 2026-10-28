# Menjalankan eksperimen 18 model di server WSL Ubuntu 22.04

Dataset penelitian terdiri dari 16 togog Bali (8 kelas, masing-masing 2 model) dan 2 slot benchmark Tanks and Temples. Semua nama dan routing dataset memakai satu katalog di `pipeline_tools/dataset_registry.py`.

## 1. Siapkan environment di server

Ikuti [SETUP_WINDOWS11.md](SETUP_WINDOWS11.md) untuk driver Windows, WSL 2, compiler, COLMAP, Conda, dan CUDA Toolkit 11.8. Gunakan salinan workspace ini beserta perubahan lokalnya, bukan clone SuGaR upstream yang belum memuat pipeline skripsi. Simpan proyek dan dataset di filesystem Linux, misalnya `/home/anggara/SuGaR`.

Repository penelitian: `https://github.com/widyaanggara/SuGaR.git`. Setelah perubahan laptop di-push, jalankan `git clone https://github.com/widyaanggara/SuGaR.git ~/SuGaR` untuk instalasi pertama. Untuk pembaruan setelahnya, hentikan pekerjaan aktif, masuk ke `~/SuGaR`, lalu jalankan `git pull --ff-only origin main`. Pull memperbarui kode; environment tetap perlu diperiksa dan extension perlu dibangun ulang bila berubah.

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

Gunakan sidebar untuk berpindah ke pengaturan, antrean, dataset, log, dan hasil evaluasi. Pengaturan dibagi menjadi tab **Resolusi foto**, **COLMAP**, **Training & antrean**, dan **LiDAR**; preset tetap terlihat di atas tab. Kolom ukuran foto acuan dapat dibuka melalui **Sesuaikan ukuran foto acuan untuk perkiraan**. Pada kartu model, buka **Lihat folder input & output** untuk melihat lokasi dataset dan hasil.

1. Pilih konfigurasi SuGaR dari repo resmi (untuk uji alur tersedia **Uji awal · low_poly + short**), lalu isi **GPU ID** sesuai indeks di `nvidia-smi`. Sesuaikan resolusi, perangkat COLMAP, matcher, fitur, refinement, dan target mesh bila diperlukan. Kolom ukuran acuan menampilkan perkiraan resolusi untuk foto 3840×2160; kolom ini hanya untuk preview, bukan mengganti ukuran input. Faktor downscale dan batas sisi diterapkan bersama: pada 2× dengan batas 1600, foto 3840×2160 menjadi sekitar 1600×900; pada 4× menjadi sekitar 960×540. Ukuran sebenarnya mengikuti foto hasil undistorsi. Kedua metode dan evaluasinya memakai fungsi pengaturan ukuran yang sama.
2. Isi foto setiap model, kemudian klik **Antrekan Semua 18 Model**. Dataset kosong dilaporkan sebagai dilewati; model yang memiliki data masuk FIFO sesuai urutan katalog.
3. Satu pekerjaan menjalankan tahap secara berurutan: ekstraksi video bila foto belum tersedia → COLMAP → 3DGS 30000 iterasi (menyimpan 7000 dan 30000) → SuGaR → evaluasi visual 3DGS → evaluasi visual SuGaR.
4. Anda bisa menambahkan pekerjaan baru ketika pekerjaan lain berjalan. Duplikat pekerjaan yang masih menunggu/berjalan tidak ditambahkan lagi. Evaluasi visual 3DGS dan SuGaR menjadi dua pekerjaan berbeda saat dijalankan lewat tombol tahap.
5. **Jeda Antrean** membiarkan pekerjaan aktif selesai lalu menahan pekerjaan berikutnya. **Stop** menghentikan seluruh pohon proses pekerjaan aktif dan menjeda antrean. Pekerjaan aktif ditandai dibatalkan; klik **Lanjutkan Antrean** untuk model berikutnya. Pekerjaan gagal dapat ditambahkan ulang.
6. Antrean disimpan di `outputs/queue/state.json`; log setiap pekerjaan di `outputs/queue/logs/<id>.log`. Klik **Log** pada baris pekerjaan untuk membuka jendela log yang diperbarui setiap dua detik. Log internal COLMAP diteruskan langsung ketika COLMAP mengeluarkan baris output, termasuk progres matching dan mapper. Pesan error pemuatan log ditampilkan dalam jendela yang sama. Sesudah server dashboard dimulai ulang, antrean dijeda. Pekerjaan yang terputus ditandai `INTERRUPTED`, bukan sukses. Pastikan tidak ada proses training lama tersisa pada GPU sebelum melanjutkan.

Konfigurasi SuGaR sekarang bersumber dari [README resmi](https://github.com/Anttwo/SuGaR#quick-start) dan [train.py resmi](https://github.com/Anttwo/SuGaR/blob/main/train.py), diperiksa pada 28 Oktober 2026. Definisi preset bersama untuk frontend/backend berada di `web_dashboard/research_presets.json` dan disajikan melalui `/api/presets`. Pilih preset secara eksplisit: konfigurasi browser lama dipertahankan sebagai **Kustom**, bukan diam-diam dianggap konfigurasi resmi.

| Konfigurasi | Parameter mesh upstream | Gaussian/segitiga | Refinement | Regularisasi |
| --- | ---: | ---: | --- | --- |
| SuGaR high_poly · long | 1000000 | 1 | long / 15000 | dn_consistency |
| SuGaR low_poly · long | 200000 | 6 | long / 15000 | dn_consistency |
| Uji awal · low_poly + short | 200000 | 6 | short / 2000 | dn_consistency |

High/low-poly serta short/medium/long adalah opsi dari upstream. Default upstream high-poly/long; konfigurasi uji awal menggabungkan opsi low-poly dan short yang tersedia, bukan rekomendasi terukur khusus RTX 4090. Medium = 7000 iterasi. Low-poly tidak otomatis lebih hemat VRAM karena menggunakan enam Gaussian per segitiga. Tidak ada preset yang menjamin setiap scene muat dalam 24 GB.

**Catatan nama parameter mesh:** README/CLI upstream menyebut `n_vertices_in_mesh`, tetapi [extractor coarse_mesh.py](https://github.com/Anttwo/SuGaR/blob/main/sugar_extractors/coarse_mesh.py) meneruskan nilai itu sebagai `decimation_target` ke Open3D `simplify_quadric_decimation`, yaitu budget segitiga pada komponen foreground/background. Karena itu UI menamainya **Parameter mesh SuGaR**. Angka preset mengikuti nilai parameter resmi; angka ini bukan jaminan jumlah vertex atau segitiga akhir gabungan. Manifest baru menyimpan nama `decimation_target`; hasil lama dengan `target_triangles` tetap dikenali untuk kompatibilitas.

Preset hanya mengubah parameter SuGaR: mesh, Gaussian/segitiga, refinement, dan regularisasi. Resolusi, GPU, COLMAP, serta kalibrasi LiDAR tetap mengikuti pilihan masing-masing. Resolusi seragam 1600 px untuk dua metode adalah adaptasi protokol penelitian proyek ini, bukan klaim default SuGaR: loader SuGaR upstream mempunyai batas 1920 px, sedangkan baseline 3DGS mempunyai aturan resize sendiri. Tetapkan satu protokol resolusi untuk seluruh perbandingan, lalu catat sebagai modifikasi metodologi. Batas COLMAP 2000 pada UI awal juga pilihan proyek, bukan rekomendasi resmi SuGaR. `0` pada batas training berarti tanpa batas tambahan setelah downscale.

`dn_consistency` merupakan rekomendasi repo yang diperbarui; menurut README, hasil paper memakai `density` untuk scene berpusat pada objek dan `sdf` untuk latar kompleks. Karena itu preset terbaru tidak boleh disebut replikasi persis tabel paper CVPR 2024. Pilihan regularisasi tersedia di tab Training; pilih sesuai protokol skripsi dan catat pilihan tersebut. Parameter tetap yang mengikuti wrapper resmi mencakup checkpoint awal 7000, `surface_level=0.3`, normal consistency 0.1, ukuran atlas tekstur 8, dan evaluasi holdout setiap kamera kedelapan. Baseline pembanding dilatih hingga 30000 iterasi.

Pengujian kontrak parameter ada di `tests/test_settings_contract.py`: nilai preset diperiksa terhadap cabang konfigurasi `train.py`, API/CLI diverifikasi meneruskan parameter, dan pemanggilan coarse/mesh/refine/export diuji dengan dependensi GPU diganti mock. Ini membuktikan routing parameter dan pencatatan konfigurasi; bukan bukti training GPU dataset penelitian telah selesai atau kualitas hasil sesuai paper. Konfigurasi antrean disimpan saat enqueue; parameter aktual SuGaR dan tautan acuan dicatat dalam log JSON serta `pipeline_result.json`. Perubahan Gaussian/segitiga atau regularisasi mengharuskan tahap SuGaR/evaluasinya dijalankan ulang, bukan memakai hasil yang berbeda konfigurasi.

Metadata acuan juga mencatat SHA-256 implementasi lokal `train.py`, extractor mesh, refinement, dan wrapper SuGaR. Hash ini mengidentifikasi file yang digunakan di fork penelitian, bukan klaim bahwa seluruh fork identik dengan upstream. Simpan commit Git eksperimen dan file konfigurasi/manifest bersama hasil skripsi.

Tombol **Periksa COLMAP** membaca build COLMAP di server melalui `/api/colmap`. **Otomatis** memakai CUDA bila header build menunjukkan dukungan CUDA, atau CPU dengan pesan yang jelas. **GPU CUDA** wajib memakai build CUDA dan gagal dengan penjelasan bila tidak tersedia; pilihan ini tidak memasang CUDA/COLMAP. Deteksi build belum membuktikan kernel GPU berhasil: lihat log saat proses berjalan. Indeks GPU fisik diterapkan melalui `CUDA_VISIBLE_DEVICES`, lalu SIFT memakai GPU logis 0. Akselerasi ini untuk ekstraksi/pencocokan SIFT; mapper masih dapat memakai CPU. [Opsi SIFT COLMAP 3.7](https://github.com/colmap/colmap/blob/3.7/src/feature/sift.h)

**Exhaustive** cocok untuk foto dengan nama bebas. **Sequential** dipakai hanya bila nama file terurut mengikuti pengambilan, misalnya frame_00001.jpg; overlap mengatur tetangga yang dicocokkan. Sequential tidak mengasumsikan nama foto acak merupakan urutan kamera. [Opsi matcher COLMAP](https://github.com/colmap/colmap/blob/3.7/src/feature/matching.h)

Pengaturan utama tersimpan di browser yang sama; skala LiDAR diisi berdasarkan kalibrasi tiap model. Seluruh pengaturan pekerjaan disalin saat tombol antrekan ditekan. Mengubah kontrol dashboard tidak mengubah pekerjaan yang sudah berjalan atau menunggu. Untuk memakai setting berbeda, batalkan pekerjaan yang menunggu lalu tambahkan ulang. Perbarui kode dan mulai ulang dashboard setelah pekerjaan aktif selesai, atau gunakan Stop bila memang ingin membatalkannya; perubahan kode tidak diterapkan ke proses yang sudah berjalan.

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

`pipeline_result.json` hanya ditulis sesudah pipeline SuGaR berhasil. Dashboard tidak menganggap folder kosong sebagai hasil selesai. Tampilan preview menggunakan render citra uji, bukan gambar atlas tekstur OBJ. Pengaturan COLMAP dicatat pada `datasets/<kelas>/<model>/colmap_settings.json`; baseline mencatat resolusi dan signature model COLMAP pada `outputs/3dgs/<kelas>/<model>/training_settings.json`. Resolusi SuGaR dan evaluasi juga dicatat dalam JSON masing-masing.

**Lewati tahap yang selesai** memeriksa file hasil, resolusi, serta konfigurasi refinement/mesh/Gaussian/regularisasi yang diminta. Mengubah resolusi training memicu pelatihan ulang dan evaluasi ulang. Mengubah batas/perangkat/fitur/matcher COLMAP memakai database SfM terpisah di `distorted/maxsize_<px>_<signature>/`, sehingga konfigurasi lama tidak tercampur; hasil undistorsi lalu dipublikasikan ke `images/` dan `sparse/0/`. Konfigurasi CPU bawaan CLI tetap memakai nama `maxsize_<px>/`. Model COLMAP lama/impor tanpa catatan resolusi masih dapat dipakai ulang pada konfigurasi historis: 3200 px, CPU, Exhaustive, 8192 fitur, overlap 10. Pesan log menjelaskan bahwa ukuran SfM-nya tidak diketahui. Default dashboard baru memakai preset Pilot (2000 px), sedangkan default CLI tetap 3200 px untuk kompatibilitas. Baseline lama tanpa `training_settings.json` akan dilatih ulang saat menjalankan Pipeline Lengkap.

Evaluasi memakai signature checkpoint untuk menghindari hasil lama setelah checkpoint diganti. Ini melanjutkan antartahap, bukan melanjutkan iterasi optimizer yang terputus. Bila dataset diubah, nonaktifkan opsi lewati dan gunakan folder dataset eksperimen baru agar model dan evaluasi lama tidak tercampur.

## 5. Evaluasi dan batas interpretasi

Pipeline lengkap menghitung PSNR, SSIM, dan LPIPS VGG pada setiap kamera kedelapan yang dikeluarkan dari pelatihan. Kedua metode memakai pembagian kamera dan resolusi evaluasi yang sama. Input LPIPS dinormalisasi ke [-1, 1] sesuai [dokumentasi resmi LPIPS](https://github.com/richzhang/PerceptualSimilarity#1-learned-perceptual-image-patch-similarity-lpips-metric). SuGaR dievaluasi sebagai representasi hybrid mesh + Gaussians; skor ini bukan skor rendering OBJ bertekstur konvensional. LPIPS dapat mengunduh bobot pretrained pada pemakaian pertama; siapkan jaringan/cache bobot di server sebelum eksperimen panjang.

Evaluasi geometri LiDAR dijalankan terpisah setelah memilih metode dan mengisi skala meter per unit model. Skala harus diukur; pipeline tidak mengasumsikan COLMAP menghasilkan meter. Ground truth harus memakai meter dan area objek yang sama. ICP memerlukan orientasi awal cukup dekat dan sekarang menolak overlap yang terlalu rendah. Selalu periksa hasil alignment sebelum memakai Chamfer di skripsi. Pusat Gaussian 3DGS dan sampel permukaan mesh SuGaR adalah representasi geometri berbeda dan perlu dijelaskan dalam metodologi.

Dashboard menampilkan delapan kartu: **Model selesai, PSNR, SSIM, LPIPS**, kemudian **Chamfer distance, Precision, Recall, F1 Score**. Gunakan filter metode untuk memilih 3DGS, SuGaR, atau keduanya; kartu metrik menampilkan rata-rata hasil terakhir yang sukses per model/metode, bukan rata-rata seluruh histori sesi. F1 dirata-ratakan dari F1 tiap model, bukan dihitung ulang dari rata-rata Precision/Recall. Jumlah hasil yang tersedia dicantumkan pada kartu; hasil yang belum dihitung tetap `—`. Model selesai mengacu pada training dan evaluasi visual kedua metode, sehingga tidak bergantung pada ketersediaan LiDAR.

Untuk menghitung geometri dari dashboard:

1. Simpan ground truth pada `datasets/<kelas>/<model>/ground_truth/lidar.ply` dalam satuan meter, dengan cakupan objek yang sama.
2. Di tab **LiDAR**, pilih metode dan isi faktor kalibrasi skala meter per unit model.
3. Klik **Evaluasi LiDAR** pada kartu model. Pekerjaan masuk antrean yang sama, terpisah dari Pipeline Lengkap yang menghitung evaluasi visual.
4. Ulangi untuk metode pembanding bila diperlukan. Pilih ambang tampilan **2 cm** atau **5 cm** pada ringkasan untuk membandingkan skor dengan ambang yang sama.

Precision adalah persentase titik rekonstruksi dengan jarak tetangga terdekat ke ground truth **< ambang**; Recall adalah persentase titik ground truth dengan jarak ke rekonstruksi **< ambang**. F1 = `2 × Precision × Recall / (Precision + Recall)`, dengan nilai 0 bila keduanya 0. Kedua ambang 2 cm dan 5 cm selalu dihitung dan diekspor; pilihan dashboard hanya mengganti tampilan. Ini metrik geometri, bukan skor klasifikasi kelas togog. [Rujukan definisi F-score Open3D](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.compute_metrics)

Chamfer proyek ini mempertahankan definisi sebelumnya: `0,5 × (mean jarak rekonstruksi→GT + mean jarak GT→rekonstruksi)`, memakai jarak Euclidean tanpa kuadrat, lalu dikonversi ke cm. Catat definisi ini dalam skripsi karena implementasi lain dapat memakai penjumlahan dua arah atau jarak kuadrat. Sampel memakai seed 0 dan maksimum 100000 titik per cloud; mesh disampel pada permukaannya. Hasil detail, jumlah titik evaluasi, transformasi ICP, signature file sumber/GT, dan faktor skala disimpan pada `outputs/<3dgs|sugar>/<kelas>/<model>/geometry/metrics.json` serta log JSON sesi. Nilai Precision/Recall/F1 pada ambang 2 cm dan 5 cm juga masuk CSV dan laporan Markdown. CSV lama diperluas secara otomatis dengan histori tetap dipertahankan; untuk mengisi metrik baru pada hasil lama, jalankan ulang evaluasi LiDAR.

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

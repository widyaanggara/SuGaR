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

Untuk memberi nama togog, buka kartu model di dashboard, klik **Ubah nama**, isi nama seperti **Ganesha**, lalu klik **Simpan nama**. Kosongkan isian untuk kembali ke nama bawaan. Nama tersimpan di server pada `datasets/model_names.json`, dibaca kembali setelah refresh/restart, dan berlaku untuk kartu model, antrean, judul log, tabel hasil, serta kolom **Nama Model** pada ekspor CSV. Nama dapat diedit tanpa menghentikan training. ID tetap `model_01`/`model_02`; jangan mengganti nama folder dataset atau output. Nama tidak mengubah pekerjaan, checkpoint, atau histori metrik. CSV mentah histori tetap dipertahankan; ekspor dashboard menambahkan nama terbaru dan ID asli. Sertakan `datasets/model_names.json` saat mencadangkan atau memindahkan dataset; metadata ini diabaikan Git agar pembaruan kode tidak menggantinya.

Untuk dua sampel Tanks and Temples, letakkan foto dari scene pilihan pada masing-masing `input/`. Bila sudah memiliki hasil COLMAP terundistorsi, masukkan `images/` dan tiga file model pada `sparse/0/`; proses dapat melewati COLMAP. Nama kedua slot mengikuti permintaan penelitian; catat nama scene resmi, sumber unduhan, dan unit ground truth dalam catatan eksperimen. Evaluasi lokal ini memakai split kamera yang sama dengan dataset Bali, bukan protokol leaderboard resmi Tanks and Temples.

## 3. Jalankan dashboard dan antrekan semua model

Untuk pratinjau desain di laptop tanpa menjalankan backend atau training, buka `web_dashboard/index.html` langsung di browser. Menu kategori, input kustom, dan perkiraan resolusi berfungsi dalam mode pratinjau lokal. Data server tidak dimuat, preset resmi dan tombol antrekan semua dinonaktifkan, dan frontend tidak mengirim permintaan API atau menjalankan polling. Untuk antrean, hasil penelitian, serta preset resmi, buka dashboard yang berjalan di server melalui langkah berikut.

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

Gunakan sidebar untuk berpindah ke pengaturan, antrean, dataset, log, dan hasil evaluasi. Pengaturan dibagi menjadi tab **Resolusi foto**, **COLMAP**, **Training**, dan **Evaluasi LiDAR**; preset tetap terlihat di atas tab. Kolom ukuran foto acuan dapat dibuka melalui **Sesuaikan ukuran foto acuan untuk perkiraan**. Pada kartu model, buka **Lihat folder input & output** untuk melihat lokasi dataset dan hasil.

Panel pengaturan menempatkan kategori di kiri dan isian kategori aktif di kanan; pada layar kecil kategori dapat digulir mendatar. Ringkasan di bawah panel membaca kontrol yang sama dengan payload pekerjaan baru, melalui `readExperimentSettings()` di dashboard. Ringkasan ini bukan konfigurasi pekerjaan yang sudah masuk antrean. Acuan repo dan catatan metodologi dapat dibuka melalui **Acuan konfigurasi dan catatan penelitian**. Navigasi kategori mendukung tombol panah atas/bawah maupun kiri/kanan, serta Home/End.

1. Pilih konfigurasi SuGaR dari repo resmi (untuk uji alur tersedia **Uji awal · low_poly + short**), lalu isi **GPU ID** sesuai indeks di `nvidia-smi`. Sesuaikan resolusi, perangkat COLMAP, matcher, fitur, refinement, dan target mesh bila diperlukan. Kolom ukuran acuan menampilkan perkiraan resolusi untuk foto 3840×2160; kolom ini hanya untuk preview, bukan mengganti ukuran input. Faktor downscale dan batas sisi diterapkan bersama: pada 2× dengan batas 1600, foto 3840×2160 menjadi sekitar 1600×900; pada 4× menjadi sekitar 960×540. Ukuran sebenarnya mengikuti foto hasil undistorsi. Kedua metode dan evaluasinya memakai fungsi pengaturan ukuran yang sama.
2. Isi foto setiap model, kemudian klik **Antrekan Model** di kanan atas, pilih semua kelas/model, lalu **Pipeline lengkap**. Popup menampilkan proses dan kesiapan model sebelum Anda menambahkan pekerjaan. Dataset kosong dilaporkan sebagai dilewati; model yang memiliki data masuk FIFO sesuai urutan katalog. Pada setiap kartu hanya ada **Antrekan model ini**: buka popup untuk memilih pipeline lengkap, COLMAP, 3DGS, SuGaR, evaluasi visual, atau evaluasi geometri. Ekstraksi video juga tersedia jika ada video pada model tersebut.
   Biarkan **Mulai pipeline → Otomatis — lanjutkan hasil terakhir tiap model** dan aktifkan **Lewati tahap yang selesai**. Model yang baru berisi foto menjalankan COLMAP; model dengan hasil COLMAP lengkap (termasuk salinan teman) langsung melanjutkan **3DGS → SuGaR → evaluasi visual kedua metode**. Jika training sudah lengkap dan cocok, pipeline melewatinya; training parsial dipulihkan dari snapshot yang cocok. Tidak perlu mengantrekan setiap tahap secara manual atau memilih awal yang sama untuk semua model.
   Model `INTERRUPTED`/gagal sebaiknya dipulihkan melalui **Antrekan Ulang** agar memakai pengaturan asli dan mendapat prioritas. Setelah restart, klik **Lanjutkan Antrean** bila antrean masih dijeda. Jika hanya ingin menerima model yang sudah memiliki hasil COLMAP, tersedia opsi **Hanya model dengan hasil COLMAP lengkap**; model tanpa hasil tersebut dilewati.
3. Satu pekerjaan menjalankan tahap secara berurutan: ekstraksi video bila foto belum tersedia → COLMAP → 3DGS 30000 iterasi (menyimpan 7000 dan 30000) → SuGaR → evaluasi visual 3DGS → evaluasi visual SuGaR.
4. Anda bisa menambahkan pekerjaan baru ketika pekerjaan lain berjalan. Duplikat pekerjaan yang masih menunggu/berjalan tidak ditambahkan lagi. Di popup evaluasi, pilih 3DGS, SuGaR, atau keduanya. Dua metode menjadi dua pekerjaan terpisah yang berjalan berurutan, bukan bersamaan. Pilihan **Evaluasi visual** dan **Evaluasi geometri** di popup memakai hasil yang sudah tersedia, tanpa menjalankan training ulang. Model/metode yang belum siap ditampilkan beserta alasannya dan dilewati.
5. **Jeda Antrean** membiarkan pekerjaan aktif selesai lalu menahan pekerjaan berikutnya. **Stop** menghentikan seluruh pohon proses pekerjaan aktif dan menjeda antrean. Pekerjaan aktif ditandai dibatalkan; klik **Lanjutkan Antrean** untuk model berikutnya. Klik **Antrekan Ulang** pada pekerjaan terputus/gagal untuk memulihkan checkpoint yang cocok dan memprioritaskannya setelah pekerjaan aktif selesai. Antrean yang dijeda tetap perlu dilanjutkan secara manual.
6. Antrean disimpan di `outputs/queue/state.json`; log setiap pekerjaan di `outputs/queue/logs/<id>.log`. Klik **Log** pada baris pekerjaan untuk membuka jendela log yang diperbarui setiap dua detik. Log internal COLMAP diteruskan langsung ketika COLMAP mengeluarkan baris output, termasuk progres matching dan mapper. Pesan error pemuatan log ditampilkan dalam jendela yang sama. Sesudah server dashboard dimulai ulang, antrean dijeda. Pekerjaan yang terputus ditandai `INTERRUPTED`, bukan sukses. Pastikan tidak ada proses training lama tersisa pada GPU sebelum melanjutkan.

Untuk hasil COLMAP impor, salin **foto hasil undistorsi** ke `datasets/<kelas>/<model>/images/` dan model kameranya ke `sparse/0/{cameras,images,points3D}.bin` (atau ketiga file `.txt`). Nama file foto harus sesuai dengan model kamera tersebut. `database.db` atau `distorted/sparse/0` saja tidak cukup untuk training; hasil perlu diundistorsi terlebih dahulu.

Mode otomatis dengan pemulihan aktif memakai `images/` dan `sparse/0` yang lengkap tanpa mencocokkan ulang pengaturan ekstraksi COLMAP. Ini menjaga hasil impor dan identitas dataset checkpoint saat retry. Pengaturan COLMAP pada panel hanya diterapkan saat membuat COLMAP baru. Untuk menghitung ulang COLMAP setelah mengganti foto atau protokol SfM, pilih **COLMAP saja** pada model tersebut, atau nonaktifkan pemulihan untuk pipeline baru. Pengaturan resolusi **training/evaluasi** tetap diperiksa dan diterapkan. Catatan `colmap_settings.json` tidak dibuat-buat atau diubah agar tampak cocok. Opsi **Hanya model dengan hasil COLMAP lengkap** (`existing_colmap`) secara eksplisit melarang COLMAP ulang: jika berkas hilang sebelum job mulai, job gagal dengan penjelasan. Pilihan awal pipeline tersimpan pada job dan dipertahankan oleh **Antrekan Ulang**. Run baru dapat memulihkan checkpoint berkala 3DGS/SuGaR yang cocok; lihat [RECOVERY.md](RECOVERY.md).

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

Pengaturan utama tersimpan di browser yang sama; satuan LiDAR serta mode penyelarasan disimpan per model di server melalui popup antrean. Seluruh pengaturan pekerjaan disalin saat tombol antrekan ditekan. Mengubah kontrol dashboard tidak mengubah pekerjaan yang sudah berjalan atau menunggu. Untuk memakai setting berbeda, batalkan pekerjaan yang menunggu lalu tambahkan ulang. Perbarui kode dan mulai ulang dashboard setelah pekerjaan aktif selesai, atau gunakan Stop bila memang ingin membatalkannya; perubahan kode tidak diterapkan ke proses yang sudah berjalan.

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

**Lewati tahap yang selesai** memeriksa file hasil, resolusi training, serta konfigurasi refinement/mesh/Gaussian/regularisasi yang diminta. COLMAP lengkap dipakai kembali, termasuk hasil impor tanpa catatan pengaturan; log memberi tahu jika pengaturan SfM lama tidak tercatat. Mengubah resolusi training memicu pelatihan ulang dan evaluasi ulang. Untuk menerapkan perubahan batas/perangkat/fitur/matcher COLMAP pada model yang sudah mempunyai COLMAP, jalankan **COLMAP saja** secara eksplisit. Proses tersebut memakai database SfM terpisah di `distorted/maxsize_<px>_<signature>/`, sehingga konfigurasi lama tidak tercampur; hasil undistorsi lalu dipublikasikan ke `images/` dan `sparse/0/`. Konfigurasi CPU bawaan CLI tetap memakai nama `maxsize_<px>/`. Default dashboard memakai preset Pilot (2000 px), sedangkan default CLI tetap 3200 px untuk kompatibilitas. Baseline lama tanpa `training_settings.json` akan dilatih ulang saat menjalankan Pipeline Lengkap.

Evaluasi memakai signature checkpoint untuk menghindari hasil lama setelah checkpoint diganti. Ini melanjutkan antartahap, bukan melanjutkan iterasi optimizer yang terputus. Bila dataset diubah, nonaktifkan opsi lewati dan gunakan folder dataset eksperimen baru agar model dan evaluasi lama tidak tercampur.

## 5. Evaluasi dan batas interpretasi

Pipeline lengkap menghitung PSNR, SSIM, dan LPIPS VGG pada setiap kamera kedelapan yang dikeluarkan dari pelatihan. Kedua metode memakai pembagian kamera dan resolusi evaluasi yang sama. Input LPIPS dinormalisasi ke [-1, 1] sesuai [dokumentasi resmi LPIPS](https://github.com/richzhang/PerceptualSimilarity#1-learned-perceptual-image-patch-similarity-lpips-metric). SuGaR dievaluasi sebagai representasi hybrid mesh + Gaussians; skor ini bukan skor rendering OBJ bertekstur konvensional. LPIPS dapat mengunduh bobot pretrained pada pemakaian pertama; siapkan jaringan/cache bobot di server sebelum eksperimen panjang.

Evaluasi geometri LiDAR dijalankan terpisah dengan satuan LiDAR yang sudah diperiksa. Dashboard mendukung pencarian skala, posisi, dan rotasi otomatis; tidak memerlukan input tinggi atau titik kaki/kepala. LiDAR dan rekonstruksi harus mencakup objek/ROI yang sama. Hasil otomatis wajib diperiksa secara visual sebelum dilaporkan dalam skripsi. Protokol, keterbatasan, dan acuan dijelaskan pada [GEOMETRY_EVALUATION.md](GEOMETRY_EVALUATION.md).

Dashboard menampilkan delapan kartu: **Model selesai, PSNR, SSIM, LPIPS**, kemudian **Chamfer distance, Precision, Recall, F1 Score**. Gunakan filter metode untuk memilih 3DGS, SuGaR, atau keduanya; kartu metrik menampilkan rata-rata hasil terakhir yang sukses per model/metode, bukan rata-rata seluruh histori sesi. F1 dirata-ratakan dari F1 tiap model, bukan dihitung ulang dari rata-rata Precision/Recall. Jumlah hasil yang tersedia dicantumkan pada kartu; hasil yang belum dihitung tetap `—`. Model selesai mengacu pada training dan evaluasi visual kedua metode, sehingga tidak bergantung pada ketersediaan LiDAR.

Untuk menghitung geometri dari dashboard:

1. Simpan PLY **point cloud LiDAR** pada `datasets/<kelas>/<model>/ground_truth/lidar.ply`. Pastikan satuan koordinat file melalui pengaturan ekspor atau pengukuran jarak yang diketahui.
2. Klik **Antrekan Model** di kanan atas, pilih semua kelas, satu kelas, atau satu model, lalu **Evaluasi geometri**. Tombol pada kartu langsung memilih model itu.
3. Pilih 3DGS, SuGaR, atau keduanya. Pilih satuan LiDAR **meter/cm/mm** per model. Satuan tidak ditebak dari ekstensi PLY.
4. Biarkan penyelarasan **Otomatis** agar sistem mencari skala, posisi, dan rotasi. Faktor skala manual hanya tersedia sebagai pilihan lanjutan. Satuan disimpan per model melalui **Simpan pengaturan LiDAR**, atau otomatis sebelum enqueue.
5. Periksa target siap dan alasan yang dilewati, lalu **Antrekan ? pekerjaan**. Metrik Chamfer, Precision, Recall, dan F1 pada kedua ambang 2/5 cm dihitung bersama per model/metode. Satu worker menjalankannya bergantian, tanpa training ulang. Pipeline lengkap tetap berakhir pada evaluasi visual.
6. Periksa overlay `aligned_reconstruction.ply` dan `ground_truth_meters.ply` di folder `outputs/<metode>/<kelas>/<model>/geometry/`. Keduanya dalam meter. Penyelarasan yang gagal pemeriksaan numerik tidak menerbitkan skor baru; lolos pemeriksaan numerik tetap perlu pemeriksaan visual untuk skripsi.

Pengaturan geometri tersimpan di `datasets/geometry_settings.json`, diabaikan Git, dan perlu disertakan saat backup/memindahkan dataset. File skala manual lama tetap didukung untuk kompatibilitas CLI; dashboard baru meminta satuan eksplisit. Pilihan disalin ke pekerjaan saat enqueue. Panduan dan protokol lengkap: [GEOMETRY_EVALUATION.md](GEOMETRY_EVALUATION.md).

Precision adalah persentase titik rekonstruksi dengan jarak tetangga terdekat ke ground truth **< ambang**; Recall adalah persentase titik ground truth dengan jarak ke rekonstruksi **< ambang**. F1 = `2 × Precision × Recall / (Precision + Recall)`, dengan nilai 0 bila keduanya 0. Kedua ambang 2 cm dan 5 cm selalu dihitung dan diekspor; pilihan dashboard hanya mengganti tampilan. Ini metrik geometri, bukan skor klasifikasi kelas togog. [Rujukan definisi F-score Open3D](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.compute_metrics)

Chamfer proyek ini mempertahankan definisi sebelumnya: `0,5 × (mean jarak rekonstruksi→GT + mean jarak GT→rekonstruksi)`, memakai jarak Euclidean tanpa kuadrat, lalu dikonversi ke cm. Catat definisi ini dalam skripsi karena implementasi lain dapat memakai penjumlahan dua arah atau jarak kuadrat. Sampel memakai seed 0 dan maksimum 100000 titik per cloud; mesh disampel pada permukaannya. Hasil detail, jumlah titik evaluasi, transformasi penyelarasan, satuan LiDAR, signature file sumber/GT, dan faktor skala disimpan pada `outputs/<3dgs|sugar>/<kelas>/<model>/geometry/metrics.json` serta log JSON sesi. Nilai Precision/Recall/F1 pada ambang 2 cm dan 5 cm juga masuk CSV dan laporan Markdown. CSV lama diperluas secara otomatis dengan histori tetap dipertahankan; untuk mengisi metrik baru pada hasil lama, jalankan ulang evaluasi LiDAR.

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

Pemulihan baru menyimpan snapshot 3DGS setiap 1.000 iterasi dan SuGaR setiap 500 iterasi, beserta optimizer/RNG. Bagian SuGaR yang selesai dicatat terpisah agar coarse/mesh/refinement tidak ikut diulang. Checkpoint lama tanpa metadata yang cukup belum tentu dapat dipulihkan otomatis. Panduan, batasan COLMAP/evaluasi, dan biaya penyimpanan: [RECOVERY.md](RECOVERY.md).

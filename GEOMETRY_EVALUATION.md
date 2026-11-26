# Evaluasi geometri terhadap LiDAR

## Dashboard

Antrean geometri dari popup dikelompokkan **satu pekerjaan per togog dan
eksperimen**. Metode yang siap dijalankan berurutan dalam satu log dengan
penanda `[1/N]` dan ringkasan status per metode di akhir. Kegagalan satu metode
tidak menghentikan metode berikutnya; pekerjaan keseluruhan berstatus FAILED
jika ada metode gagal, sementara skor metode yang berhasil tetap tersimpan.
Run All melewati evaluasi yang masih valid berdasarkan signature rekonstruksi dan
LiDAR, satuan, mode alignment, faktor skala manual, pilihan ICP, dan versi
algoritma otomatis. Hasil baseline dan Scaffold diperiksa terpisah. Pemeriksaan
diulang ketika pekerjaan kelompok berjalan, sehingga retry hanya menjalankan
metode yang belum memiliki hasil valid, tanpa training ulang. Antrean lama per metode tetap didukung; buat
pekerjaan baru dari popup untuk memakai pengelompokan. API mengaktifkannya
dengan `group_geometry: true`.

Evaluasi geometri menyediakan pilihan **Hasil yang dievaluasi: Baseline / Ablasi
Scaffold**, lalu checkbox 3DGS, SuGaR, 2DGS, dan RaDe-GS. Tombol **Pilih semua
metode** memilih keempatnya. Status sumber di baris LiDAR mengikuti eksperimen
terpilih; 2DGS dan RaDe-GS memerlukan mesh hasil ekstraksi, bukan checkpoint saja.
Ground truth tetap sama di folder dataset. Hasil Scaffold disimpan terpisah pada
`outputs/ablations/scaffold_anchor/<metode>/<kelas>/<model>/geometry/`, dan label
rekap metrik memakai akhiran `_scaffold_anchor`. CLI mendukung
`--experiment scaffold_anchor` (default `baseline`). Evaluasi ini tidak melatih
ulang Scaffold atau memerlukan environment training Scaffold.

1. Simpan point cloud **hasil LiDAR** dari SLAM GO POST, bukan model 3DGS dari software itu, pada `datasets/<kelas>/<model>/ground_truth/lidar.ply`.
2. Klik **Antrekan Model** di kanan atas. Pilih semua kelas, satu kelas, atau satu model. Tombol **Antrekan model ini** pada kartu langsung memilih model tersebut.
3. Pilih **Evaluasi geometri**, lalu 3DGS, SuGaR, atau keduanya.
4. Pilih satuan **koordinat file LiDAR** per model: meter (`m`), sentimeter (`cm`), atau milimeter (`mm`). Sistem tidak menebak satuan dari ekstensi PLY. Periksa satuan ekspor atau bandingkan jarak pada file dengan ukuran nyata yang diketahui.
5. Biarkan **Otomatis — cari skala, posisi, dan rotasi**. Tidak perlu mengetahui unit rekonstruksi, mengisi tinggi patung, atau memilih titik kaki/kepala.
6. Periksa pekerjaan siap dan alasan target yang dilewati, lalu klik **Antrekan … pekerjaan**. Satu worker menjalankannya bergantian. Semua empat metrik dan kedua ambang 2/5 cm dihitung bersama per metode.

**Simpan pengaturan LiDAR** hanya menyimpan pilihan; tidak memulai evaluasi. Pilihan tersimpan per model dalam `datasets/geometry_settings.json` (diabaikan Git; sertakan dalam backup dataset). Saat enqueue, seluruh pilihan disalin ke job. Mengubah pilihan berikutnya tidak mengubah job yang sudah menunggu.

Pipeline lengkap berakhir pada evaluasi visual. Evaluasi geometri dijalankan terpisah, tanpa COLMAP atau training ulang. Checkpoint atau mesh yang diperlukan harus sudah tersedia.

## Cakupan objek dan keterbatasan

Jika pemeriksaan overlap otomatis menolak kandidat, log menampilkan overlap dua
arah dan evaluasi tidak menerbitkan skor. Kandidat yang ditolak dan LiDAR dalam
meter disimpan pada `geometry/alignment_failures/attempt_*/` sebagai
`rejected_candidate.ply`, `ground_truth_meters.ply`, dan `diagnostics.json`.
Buka kedua PLY bersama untuk diagnosis; kandidat ini belum merupakan alignment
yang sah untuk pelaporan skor. Setiap percobaan memakai folder terpisah dan tidak
menimpa metrik maupun overlay sukses sebelumnya.

Log server memuat `[KONFIGURASI]`, `[INPUT]`, `[UKURAN]`, `[SATUAN]`,
`[SAMPLING]`, progres `[ALIGN 1/3]` sampai `[ALIGN 3/3]`, dan hasil overlap
dua arah. `[ALIGN SYARAT]` membedakan ambang penerimaan registrasi dari ambang
metrik 2/5 cm. Saat gagal, salin log mulai `[INPUT 1/2]` beserta
`diagnostics.json` jika tersedia untuk diagnosis jarak jauh. Error teknis selain
penolakan overlap menyertakan traceback; tidak semua kegagalan menghasilkan
kandidat overlay (misalnya input kosong atau file gagal dibaca).

LiDAR dan rekonstruksi harus mencakup **patung/ROI yang sama**. Rekonstruksi seluruh bangunan tidak dapat langsung dianggap sepadan dengan scan patung saja. Siapkan crop yang konsisten untuk eksperimen; implementasi ini tidak melakukan segmentasi semantik patung otomatis. Jangan mengganti file keluaran Gaussian asli dengan PLY biasa hasil crop: atribut Gaussian diperlukan untuk rendering. Model yang terhalang, sangat tidak lengkap, atau simetris dapat memerlukan penyelarasan manual.

Pencarian otomatis bisa gagal atau menghasilkan orientasi yang salah pada objek simetris. Lolos pemeriksaan overlap numerik **bukan** verifikasi kebenaran semantik. Periksa overlay sebelum melaporkan skor skripsi. Jika pemeriksaan numerik gagal, proses gagal tanpa menerbitkan skor sesi baru; histori sukses sebelumnya dipertahankan.

Mode **Lanjutan — gunakan faktor skala manual** tetap tersedia. Pilih satuan LiDAR dan berikan faktor rekonstruksi-ke-meter yang sudah diketahui. ICP memperbaiki posisi/rotasi; nonaktifkan hanya untuk data yang sudah sejajar. Nilai manual tidak diasumsikan `1`.

## Protokol otomatis yang diterapkan

Versi algoritma: `sim3_pca_fpfh_trimmed_v1`, implementasi khusus proyek, bukan reproduksi persis evaluator Tanks and Temples atau preset resmi SuGaR.

- GT dikonversi ke meter: `m × 1`, `cm × 0,01`, `mm × 0,001`, terhadap origin. Data sumber asli tidak ditimpa.
- Maksimum 100.000 titik evaluasi per cloud, seed 0. Mesh SuGaR disampling pada permukaan; 3DGS memakai pusat Gaussian, bukan permukaan Gaussian atau isosurface. Perbedaan representasi ini harus dicatat dalam metodologi.
- Registrasi memakai maksimal 2.500 titik per cloud, median dan persentil radius 90% untuk normalisasi. Titik radius terjauh 1% hanya dikeluarkan dari inisialisasi registrasi; sampel evaluasi tetap menyertakannya.
- Kandidat orientasi: identitas, 24 rotasi PCA tanpa refleksi, dan FPFH/RANSAC Open3D. Voxel ternormalisasi 0,05; radius normal 0,15; FPFH 0,3; RANSAC 50.000 iterasi, confidence 0,999, empat korespondensi, ambang 0,15. Seed 0 digunakan; paralelisme Open3D dapat memengaruhi pengulangan RANSAC.
- Setiap kandidat diperbaiki dengan korespondensi tetangga terdekat dua arah, trim 20% jarak terbesar, dan fitting Umeyama. Tiga kandidat terbaik diperbaiki sampai 100 iterasi. Skala ternormalisasi dibatasi 0,2–5. Transformasi akhir hanya satu skala seragam, rotasi, dan translasi, bukan deformasi nonrigid.
- Pemeriksaan registrasi: minimal 35% titik pada **kedua arah** berada dalam jarak ternormalisasi 0,08. Ambang ini adalah pemeriksaan implementasi proyek, bukan ambang skor 2/5 cm atau rekomendasi resmi benchmark. Skor metrik tidak dipakai untuk mengubah ambang ini per objek.
- Penyelarasan ini menghilangkan kebebasan skala SfM sebelum menilai bentuk. Skor tidak mengukur kesalahan skala absolut rekonstruksi sebelum penyelarasan.

## Metrik dan hasil yang disimpan

Chamfer (cm) = `100 × 0,5 × (mean jarak sumber→GT + mean jarak GT→sumber)`, jarak Euclidean **tanpa kuadrat**. Precision/Recall menggunakan jarak **strict < 0,02 m / 0,05 m**. F1 adalah harmonic mean P/R; semuanya dilaporkan persen. Sampling dan ambang harus sama untuk seluruh objek dan kedua metode.

Setiap metode menyimpan pada `outputs/<3dgs|sugar>/<kelas>/<model>/geometry/`:

- `metrics.json`: skor, satuan GT, estimasi skala, transformasi sumber asli→GT meter, versi algoritma, overlap registrasi, seed, dan signature file sumber/GT.
- `aligned_reconstruction.ply`: sampel rekonstruksi yang sudah sejajar dalam meter.
- `ground_truth_meters.ply`: sampel LiDAR dalam meter. Buka dua PLY ini bersama di CloudCompare untuk memeriksa overlay, termasuk jika file LiDAR asal bersatuan cm/mm.

Skor juga masuk log sesi serta CSV/Markdown riwayat. Status dashboard memeriksa kesesuaian file dan pilihan terhadap laporan terakhir. Mengubah satuan atau mengganti berkas menyebabkan hasil perlu dihitung ulang. File kalibrasi lama `datasets/geometry_scales.json` dan CLI manual tetap didukung; dashboard baru meminta satuan eksplisit dan tidak diam-diam memakai asumsi meter dari file lama.

Contoh CLI otomatis setelah satuan dikonfirmasi:

```bash
python pipeline_tools/05_evaluate_comparison.py \
  --statue_class Dewa --model_id model_01 --method SuGaR \
  --model_path outputs/sugar/Dewa/model_01/textured_mesh/model.obj \
  --gt_lidar datasets/Dewa/model_01/ground_truth/lidar.ply \
  --lidar_unit m --auto_align
```

Ganti path mesh dengan hasil aktual. `--auto_align` tidak boleh digabung dengan `--scale_factor` atau `--no_icp`. CLI lama tanpa `--lidar_unit` tetap menganggap meter demi kompatibilitas; dashboard baru tidak menggunakan default diam-diam tersebut.

## Acuan

- [Open3D 0.17: RANSAC berdasarkan fitur](https://www.open3d.org/docs/0.17.0/python_api/open3d.pipelines.registration.registration_ransac_based_on_feature_matching.html).
- [Open3D 0.17: estimasi transformasi dengan skala](https://www.open3d.org/docs/0.17.0/python_api/open3d.pipelines.registration.TransformationEstimationPointToPoint.html).
- [Paper Tanks and Temples, bagian evaluasi](https://vladlen.info/papers/tanks-and-temples.pdf): penyelarasan similarity sebelum evaluasi. Benchmark tersebut juga menggunakan pose kamera dan aturan cropping/resampling tersendiri; implementasi proyek ini berbeda.

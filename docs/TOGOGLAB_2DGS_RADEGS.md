# Menjalankan 2DGS dan RaDe-GS dari TogogLab

Keempat metode memakai antrean tunggal dashboard, dataset bersama, serta output terpisah.
Pipeline lengkap baru menjalankan satu patung hingga selesai: COLMAP → 3DGS → SuGaR →
2DGS → RaDe-GS → evaluasi visual dan mesh semua metode, baru worker mengambil patung
berikutnya. Pilih sendiri patung yang ingin dijalankan; tidak perlu menghapus hasil yang
sudah benar. Job pipeline lama yang telah tersimpan tetap memakai cakupan lamanya.

## Persiapan server

Jalankan dari root TogogLab jika repository belum ada:

```bash
git clone --branch feature/scaffold-anchor-init --recurse-submodules https://github.com/Nuuasli/2DGS.git external/2DGS
git clone --branch thesis-development --recurse-submodules https://github.com/Pandecod/RaDe-GS.git external/RaDe-GS
python pipeline_tools/init_dataset_folders.py
```

Clone lokal di `external/` belum terdaftar sebagai submodule TogogLab. Mengunduh kode
TogogLab saja tidak mengunduh kedua repo tersebut. Folder output dibuat saat inisialisasi.

Siapkan environment terpisah mengikuti dokumentasi masing-masing repository, atau
gunakan environment teman yang sudah berhasil menjalankan branch tersebut. Di setiap
environment, periksa:

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

Salin path `sys.executable` untuk diisi di dashboard. PyTorch, rasterizer native, serta
dependensi rendering/mesh/evaluasi harus terpasang pada environment metode. Environment
dashboard tetap perlu dependensi evaluasi geometri TogogLab untuk evaluasi LiDAR.

## Penggunaan dashboard

1. Restart dashboard setelah memperbarui kode, lalu refresh browser.
2. Siapkan `datasets/<kelas>/<model_id>/images/` dan `sparse/0/`. Jika belum ada,
   jalankan Rekonstruksi COLMAP dari kartu patung terlebih dahulu.
3. Klik **Antrekan Model** atau **Antrekan Kelas**, pilih **Pipeline lengkap**, lalu
   **Baseline 3DGS + SuGaR + 2DGS + RaDe-GS**. Isi Python 2DGS dan Python RaDe-GS
   pada server. Ini membuat satu job lengkap per patung, bukan empat job terpisah.
   Opsi **2DGS + evaluasi** atau **RaDe-GS + evaluasi** tetap tersedia untuk satu metode.
4. Isi path Python environment metode pada server dan pilih downscale gambar.
   Jika kosong, proses menggunakan Python dashboard. Pengaturan training SuGaR tidak
   diterapkan ke dua metode ini.
5. Klik **Antrekan**. Antrean berjalan bergantian, dan tombol Stop menghentikan pohon
   proses melalui supervisor dashboard yang sama.

Alur 2DGS: training 30.000 iterasi → rendering dan mesh native → PSNR/SSIM/LPIPS.
Alur RaDe-GS: training 30.000 iterasi → rendering → PSNR/SSIM/LPIPS → mesh native.
Keduanya memakai split `--eval` native repository; pastikan split dan resolusi cocok
sebelum membandingkan eksperimen penelitian.

Dengan opsi melanjutkan hasil aktif, checkpoint 30.000 iterasi beserta `cfg_args`
dipakai kembali tanpa training ulang. Jika belum lengkap, checkpoint `chkpnt*.pth`
terakhir yang sesuai digunakan untuk pemulihan. Rendering/evaluasi lengkap dari runner
TogogLab dilewati jika receipt dan checkpoint masih cocok. Hasil impor tanpa receipt
dievaluasi ulang ketika diantrekan. Bila salah satu tahap gagal, job patung berstatus
FAILED; worker mengikuti kebijakan antrean yang ada dan melanjutkan job berikutnya.
Gunakan Antrekan Ulang untuk memulihkan patung yang gagal.

## Ablasi empat metode

Pilih **Ablasi Scaffold: 3DGS + SuGaR + 2DGS + RaDe-GS**, lalu isi repository dan
Python Scaffold-GS. Setelah COLMAP, Scaffold-GS menghasilkan satu point cloud anchor
berwarna yang dipakai bersama untuk inisialisasi 3DGS dan 2DGS/RaDe-GS. SuGaR memakai
hasil 3DGS ablasi. 2DGS menerima `--init_ply`; RaDe-GS membaca anchor melalui salinan
dataset pada folder `initialization/` di outputnya. COLMAP sumber tidak ditimpa.

Output ablasi dan hasil evaluasi terpisah dari baseline:

```text
outputs/ablations/scaffold_anchor/3dgs/<kelas>/<model_id>/
outputs/ablations/scaffold_anchor/sugar/<kelas>/<model_id>/
outputs/ablations/scaffold_anchor/2dgs/<kelas>/<model_id>/
outputs/ablations/scaffold_anchor/radegs/<kelas>/<model_id>/
```

Keempat metode tampil di tab Studi ablasi, ekspor Excel, serta perbandingan
ablasi-minus-baseline. Evaluasi geometri LiDAR tetap merupakan proses terpisah
dengan pengaturan satuan dan alignment, seperti sebelumnya.

Untuk evaluasi visual tanpa training, pilih **Evaluasi visual**, lalu metode
**2DGS saja** atau **RaDe-GS saja** dan isi Python yang sesuai. Evaluasi geometri
dijalankan terpisah melalui opsi yang sama seperti metode lama: pilih metode,
satuan LiDAR, dan pengaturan alignment. Pilihan **3DGS dan SuGaR** tetap hanya
mengevaluasi kedua metode tersebut.

## Penyimpanan

```text
outputs/2dgs/<kelas>/<model_id>/
outputs/radegs/<kelas>/<model_id>/
```

Checkpoint: `point_cloud/iteration_30000/point_cloud.ply` dan `cfg_args`.
Metrik visual native: `results.json`, key `ours_30000` dengan `PSNR`, `SSIM`, `LPIPS`.
Preview: `test/ours_30000/renders/`.
Mesh 2DGS: `train/ours_30000/fuse_post.ply`.
Mesh RaDe-GS: `recon_post.ply`.

Hasil native yang dipindahkan ke struktur ini dapat ditampilkan tanpa training ulang.
Rekap dari proses TogogLab dicatat di CSV bersama dan masuk ekspor Excel.
Tombol hapus patung juga mencakup output kedua metode ini.

## Batas verifikasi

Pemeriksaan awal di server, tanpa training:

```bash
python pipeline_tools/check_method_environments.py \
  --python_2dgs /home/wisnu/miniconda3/envs/2dgs/bin/python \
  --python_radegs /home/pande/miniconda3/envs/radegs/bin/python \
  --scaffold_python /path/ke/environment/scaffold/bin/python \
  --scaffold_repo external/Scaffold-GS
```

Ganti path Python Scaffold dengan path sebenarnya. Jalankan sebagai user yang
menjalankan dashboard. Pemeriksaan mencakup akses executable, PyTorch/CUDA, import
entry point training/rendering/evaluasi/mesh native melalui `--help`, serta adapter
Scaffold yang benar-benar digunakan TogogLab. Kode keluar 1 berarti ada pemeriksaan
yang gagal; detailnya dicetak. Lulus pemeriksaan ini belum menguji training, isi
dataset, ataupun kecukupan VRAM saat training/ekstraksi mesh.

Integrasi diuji dengan tes CPU untuk routing command, antrean, checkpoint, artifact,
penolakan hasil evaluasi yang gagal, tabel metrik, geometri, dan ekspor. Tes tersebut
bukan bukti keberhasilan training GPU. Jalankan satu patung melalui environment server
yang benar untuk memverifikasi dependensi CUDA dan kecukupan memori perangkat.

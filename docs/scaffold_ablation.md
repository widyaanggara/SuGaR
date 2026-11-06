# Ablasi inisialisasi anchor Scaffold-GS

Alur: foto + COLMAP yang sudah ada → training Scaffold-GS → ekspor anchor dengan
warna titik COLMAP terdekat → training vanilla 3DGS → SuGaR → evaluasi visual.
Ini eksperimen **inisialisasi**, bukan pemindahan seluruh model Scaffold ke SuGaR.
Offset, fitur neural, opacity, dan atribut yang bergantung pandangan tidak ditransfer.
Anchor juga bukan jaminan titik permukaan yang lebih akurat. Duplikat koordinat
dihapus; koordinat tidak diubah skala atau sistem acuannya.

## Persiapan

Kode [Scaffold-GS resmi](https://github.com/city-super/Scaffold-GS) dicatat sebagai
Git submodule di `external/Scaffold-GS`, dengan commit yang dipatok oleh proyek.
Dependency CUDA-nya tetap perlu diinstal pada environment Conda terpisah.
Gunakan Linux/WSL sesuai environment training proyek.
Environment pipeline memerlukan numpy, scipy, dan plyfile untuk konversi.
Foto undistorsi dan cameras/images/points3D COLMAP tetap di dataset asli.

Untuk memperbarui salinan proyek yang sudah ada di server (hentikan pekerjaan
aktif dan dashboard sebelum memperbarui kode):

```bash
cd ~/SuGaR
git pull --ff-only origin main
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

Untuk clone baru, gunakan `git clone --recurse-submodules
https://github.com/widyaanggara/SuGaR.git ~/SuGaR` dalam satu baris.
`git pull` saja belum cukup untuk mengambil isi submodule. Jangan menjalankan
`git pull` tersendiri di dalam Scaffold untuk eksperimen yang hendak memakai
versi yang dipatok proyek.

Instalasi pertama environment Scaffold, dari terminal Ubuntu/WSL:

```bash
cd ~/SuGaR/external/Scaffold-GS
conda env create --file environment.yml
conda run -n scaffold_gs python -c "import sys; print(sys.executable)"
```

Jangan membuat environment ulang jika sudah tersedia. Environment upstream
memakai Python 3.7, PyTorch 1.12.1, dan CUDA runtime 11.6; kecocokan GPU,
CUDA Toolkit/compiler, dan extension tetap perlu diuji di server.
Pemasangan submodule tidak menginstal dependency atau menjalankan training.

Pada dashboard isi **Folder repository Scaffold-GS** dengan hasil `pwd` di
folder tersebut, misalnya `/home/anggara/SuGaR/external/Scaffold-GS`, dan
**Python environment Scaffold-GS** dengan executable dari perintah di atas.
Jalankan dashboard dari environment `sugar` seperti biasa.

```bash
python pipeline_tools/batch_run_all.py --class_name Dewa --model_id model_01 \
  --skip_colmap --experiment scaffold_anchor \
  --scaffold_repo /home/user/SuGaR/external/Scaffold-GS \
  --scaffold_python /home/user/miniconda3/envs/scaffold_gs/bin/python \
  --scaffold_iterations 30000 --scaffold_resolution 1
```

Hilangkan `--model_id` untuk satu kelas, atau kedua filter untuk seluruh katalog.
Batch memproses patung berurutan dan melanjutkan patung berikutnya jika satu gagal;
exit code akhirnya bukan nol jika ada kegagalan. Antrean dashboard menggunakan
worker tunggal yang sama: Pipeline lengkap → eksperimen inisialisasi Scaffold →
isi folder repo dan Python pada komputer training → antrekan.

Baseline tetap memakai `outputs/3dgs` dan `outputs/sugar`. Eksperimen memakai
`outputs/ablations/scaffold_anchor/{scaffold,3dgs,sugar}/<kelas>/<model>`.
Log/CSV diberi metode `3DGS_scaffold_anchor` dan `SuGaR_scaffold_anchor`.
Kartu status/preview baseline pada dashboard tetap menunjuk baseline; lihat folder
ablasi dan log antrean untuk artefak eksperimen. Evaluasi visual otomatis sudah
diarahkan ke output ablasi; evaluasi geometri manual dashboard masih untuk baseline.

## Pemulihan dan reproduksibilitas

Tahap Scaffold yang selesai dipakai ulang hanya jika identitas foto/COLMAP,
kode, dan konfigurasi cocok serta file output masih cocok dengan receipt.
Scaffold yang terputus **diulang dari awal**; checkpoint optimizer upstream belum
dipakai untuk pemulihan. Training 3DGS dan SuGaR memakai mekanisme recovery proyek.
`--no_resume` membuat run Scaffold baru. Input PLY turut mengikat recovery 3DGS.
Jangan menjalankan dua supervisor/antrean berbeda untuk output yang sama.

Launcher memanggil fungsi training upstream, menonaktifkan pemilihan GPU otomatis
upstream dan koneksi viewer, serta melewati rendering/evaluasi Scaffold setelah
training. GPU mengikuti antrean; `--eval` mempertahankan split uji upstream.
Scaffold memakai downscale tersendiri (`1/2/4/8`), sedangkan training 3DGS/SuGaR
tetap mengikuti `image_downscale` dan `image_max_size`; catat keduanya saat melapor.
Metadata sumber, konfigurasi, jumlah anchor, dan waktu Scaffold tersimpan per run.
Tambahkan biaya training Scaffold ketika membandingkan total waktu metode.

## Validasi

Tes CPU mencakup konversi koordinat/warna, input invalid, reuse dan invalidasi,
urutan pipeline, penghentian downstream saat gagal, serta isolasi antrean baseline.
Uji GPU end-to-end satu patung tetap diperlukan di environment Scaffold-GS sebelum
menjalankan seluruh dataset. Belum ada klaim peningkatan kualitas rekonstruksi.

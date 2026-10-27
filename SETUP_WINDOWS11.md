# Panduan Setup SuGaR di WSL Ubuntu 22.04

**Riset skripsi: komparasi 3DGS vs SuGaR pada 16 togog Bali dan 2 sampel Tanks and Temples**

Panduan ini untuk server Windows dengan Ubuntu 22.04 yang berjalan di WSL 2. Gunakan spesifikasi GPU server yang benar-benar ditampilkan oleh `nvidia-smi`; GPU laptop lokal tidak menjadi acuan. Untuk struktur lengkap 18 dataset, antrean dashboard, dan akses dari komputer lain, lihat [panduan operasional server](RUN_SERVER.md).

## 1. Masuk ke akun kerja Ubuntu

Buka aplikasi Ubuntu. Berpindah ke akun kerja yang sudah disiapkan:

```bash
su anggara
```

Masukkan password akun Linux saat diminta. Setelah berhasil, pindah ke home akun tersebut:

```bash
cd ~
whoami
pwd
```

`whoami` seharusnya menampilkan `anggara`, dan `pwd` biasanya menampilkan `/home/anggara`. Jalankan perintah proyek setelah masuk ke akun ini dan berada di home-nya.

## 2. Pastikan WSL, Ubuntu, dan GPU siap

Dari PowerShell Windows, pastikan distro Ubuntu menggunakan WSL 2:

```powershell
wsl --list --verbose
```

Distro `Ubuntu-22.04` harus menunjukkan versi `2`. Lalu buka terminal Ubuntu dan jalankan:

```bash
lsb_release -a
uname -r
nvidia-smi
```

`lsb_release` seharusnya menunjukkan Ubuntu 22.04. `nvidia-smi` harus menampilkan GPU NVIDIA pada server. Pasang atau perbarui driver NVIDIA di Windows jika GPU belum terlihat. **Jangan memasang driver NVIDIA Linux di dalam WSL**; WSL menggunakan driver dari Windows.

## 3. Siapkan Ubuntu

Pasang compiler C++, Git, COLMAP, dan pustaka sistem yang diperlukan:

```bash
sudo apt update
sudo apt install -y build-essential cmake git wget curl ffmpeg colmap libgl1 libglib2.0-0
```

### Masuk ke folder proyek

Setelah `cd ~`, lihat folder kerja yang tersedia:

```bash
ls -la
```

Masuk ke folder SuGaR yang disediakan di home, misalnya:

```bash
cd ~/SuGaR
```

Pastikan folder tersebut berisi `README.md`, `environment.yml`, dan `pipeline_tools/`:

```bash
ls
```

Jika nama atau lokasi folder yang disiapkan berbeda, gunakan path yang diberikan teman Anda. Jalankan perintah instalasi dari folder utama proyek ini.

## 4. Pasang Conda dan environment proyek

Jika Conda belum tersedia di Ubuntu, pasang Miniconda:

```bash
cd "$HOME"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
```

Ikuti installer, izinkan `conda init`, lalu tutup dan buka lagi terminal Ubuntu. Masuk ke folder proyek dan buat environment dari file proyek:

```bash
conda --version
conda env list
```

Jika environment `sugar` belum ada, buat dari file proyek:

```bash
conda env create -f environment.yml
```

Environment ini mengunci kombinasi yang dipakai kode: Python 3.9, PyTorch 2.0.1, CUDA 11.8, dan PyTorch3D 0.7.4. Jangan menaikkan versi PyTorch/CUDA secara terpisah dulu karena extension CUDA proyek dibangun terhadap kombinasi ini.

Tambahkan CUDA Toolkit 11.8 ke environment agar `nvcc` tersedia untuk kompilasi extension. Paket Conda ini memasang toolkit di environment dan tidak memasang driver Linux ke WSL:

```bash
conda install -n sugar nvidia/label/cuda-11.8.0::cuda-toolkit -y
conda activate sugar
```

OpenCV sudah tercantum di `environment.yml`. Jika memakai environment lama, tambahkan paket ekstraksi video dan build:

```bash
python -m pip install opencv-python-headless==4.10.0.84
python -m pip install ninja==1.11.1.1
```

Pastikan versi compiler dan PyTorch terbaca:

```bash
nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU tersedia:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU tidak terdeteksi')"
```

`GPU tersedia` harus bernilai `True` sebelum melanjutkan kompilasi extension.

## 5. Pasang extension CUDA SuGaR

Jalankan dari folder utama proyek dengan environment `sugar` aktif:

```bash
python -m pip install --no-build-isolation -e ./gaussian_splatting/submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation -e ./gaussian_splatting/submodules/simple-knn
```

Nvdiffrast bersifat opsional. Untuk instalasi awal, lewati paket ini dan gunakan PyTorch3D. Jika diperlukan, gunakan versi terkunci berikut; jangan memasang branch terbaru ke environment PyTorch 2.0.1:

```bash
python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3'
```

Kode menggunakan backend CUDA Nvdiffrast, sehingga tidak memerlukan konteks OpenGL. Untuk memaksa PyTorch3D, jalankan `export SUGAR_RASTERIZER=pytorch3d` sebelum memulai pipeline.

## 6. Siapkan folder data dan hasil

```bash
bash setup_workspace.sh
```

Letakkan data setiap patung dengan susunan seperti berikut:

```text
datasets/Dewa/model_01/input/        # foto asli untuk COLMAP
datasets/Dewa/model_01/images/       # hasil undistorsi COLMAP (otomatis)
datasets/Dewa/model_01/raw_video/    # video opsional
datasets/Dewa/model_01/ground_truth/ # point cloud LiDAR opsional
```

## 7. Jalankan pipeline satu model

Contoh berikut menjalankan seluruh tahap untuk `Dewa/model_01`.

### Ekstrak frame video (opsional)

```bash
python pipeline_tools/01_extract_frames.py \
  -v datasets/Dewa/model_01/raw_video/DJI_001.MP4 \
  -o datasets/Dewa/model_01/input \
  --fps 2.0 --statue_class Dewa --model_id model_01
```

### Jalankan COLMAP

```bash
python pipeline_tools/02_run_colmap.py \
  -s datasets/Dewa/model_01 --statue_class Dewa --model_id model_01
```

COLMAP memakai CPU secara default agar kompatibel dengan paket Ubuntu yang tidak menyertakan dukungan CUDA. Jika binary COLMAP Anda mendukung CUDA, tambahkan `--use_gpu`. Pelatihan 3DGS dan SuGaR tetap memakai GPU.

### Latih Vanilla 3DGS dan SuGaR

```bash
python pipeline_tools/03_train_3dgs.py \
  -s datasets/Dewa/model_01 --statue_class Dewa --model_id model_01 --iterations 30000

python pipeline_tools/04_train_sugar.py \
  -s datasets/Dewa/model_01 \
  -c outputs/3dgs/Dewa/model_01 \
  --statue_class Dewa --model_id model_01 --refinement_time short
```

Hasil 3DGS berada di `outputs/3dgs/Dewa/model_01/`, sedangkan hasil SuGaR berada di `outputs/sugar/Dewa/model_01/`.

### Evaluasi dengan point cloud LiDAR (opsional)

```bash
python pipeline_tools/05_evaluate_comparison.py \
  --statue_class Dewa --model_id model_01 --method SuGaR \
  --model_path outputs/sugar/Dewa/model_01/textured_mesh/<nama-file-obj-hasil>.obj \
  --gt_lidar datasets/Dewa/model_01/ground_truth/lidar.ply \
  --scale_factor 1.0
```

Ganti nama file dengan OBJ yang benar-benar dihasilkan di folder `textured_mesh/`. Ganti `1.0` dengan faktor kalibrasi yang diperoleh dari ukuran nyata objek dan ukuran model. Jangan gunakan nilai tanpa kalibrasi; skala rekonstruksi COLMAP dari satu kamera bergerak tidak otomatis sama dengan meter.

## 8. Jalankan dashboard

Dengan environment `sugar` aktif, jalankan:

```bash
bash start_dashboard.sh 8080
```

Buka `http://localhost:8080` di browser Windows. Hentikan server dengan `Ctrl+C` di terminal Ubuntu.

Ringkasan evaluasi tersimpan di `outputs/logs/evaluation_reports/summary_comparison.csv` dan `summary_comparison.md`.

Dashboard sekarang mendukung 18 model dan antrean persisten satu worker. Pipeline lengkap menjalankan ekstraksi opsional, COLMAP, 3DGS, SuGaR, serta evaluasi PSNR/SSIM/LPIPS kedua metode. Evaluasi LiDAR tetap opsional dan memerlukan skala terkalibrasi. Ikuti [RUN_SERVER.md](RUN_SERVER.md) untuk menjalankan semua model dan memahami jeda/lanjut/stop.

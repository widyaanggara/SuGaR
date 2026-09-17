# Panduan Setup & Eksekusi Workspace di Windows 11 Pro
**Riset Skripsi: Komparasi 3DGS vs SuGaR pada 8 Kelas Patung Bali**

Dokumen ini disusun khusus untuk spesifikasi PC Anda:
* **OS**: Windows 11 Pro
* **CPU**: AMD Ryzen 7 8700G (8 Cores, 16 Threads)
* **GPU**: NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)
* **RAM**: 64 GB
* **Storage**: Samsung SSD 990 Pro 1 TB

---

## 1. Persiapan Struktur Folder (Langkah Pertama)

Sebelum memulai, jalankan skrip inisialisasi folder di terminal (atau *double-click* file `setup_workspace.bat` di Windows Explorer):
```bash
python pipeline_tools/init_dataset_folders.py
```
Perintah ini akan membuat folder untuk seluruh 8 kelas patung di `datasets/` dan folder hasil di `outputs/`.

---

## 2. Pilihan Instalasi Lingkungan (Environment)

### OPSI A: Menggunakan WSL 2 Ubuntu 22.04 (SANGAT DIREKOMENDASIKAN ⭐⭐⭐⭐⭐)
*Mengapa WSL2? Karena seluruh pustaka grafika 3D (PyTorch3D, nvdiffrast, CUDA rasterizer) dirancang native untuk Linux. Driver NVIDIA Windows 11 otomatis mengekspos GPU RTX 4060 Ti ke dalam WSL2 tanpa instalasi driver tambahan, serta 100% bebas dari galat kompilator Visual Studio MSVC.*

#### Langkah A1: Install WSL 2 (Sekali Saja)
1. Buka **PowerShell** sebagai Administrator di Windows 11.
2. Jalankan perintah:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
3. Restart PC jika diminta, lalu buka aplikasi **Ubuntu 22.04** dari Start Menu dan buat *username* serta *password*.

#### Langkah A2: Akses Folder Project dari Ubuntu
Di dalam terminal Ubuntu, folder project Anda di Windows dapat diakses langsung via `/mnt/c/`:
```bash
cd "/mnt/c/Users/anggara/Documents/Mata Kuliah Kampus/Mata Kuliah Semester 7/Skripsi/Project/SuGaR"
```

#### Langkah A3: Install Miniconda di Ubuntu WSL
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

#### Langkah A4: Buat Environment Conda `sugar`
```bash
# 1. Buat environment Python 3.9
conda create -n sugar python=3.9 -y
conda activate sugar

# 2. Install PyTorch 2.0.1 dengan CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install PyTorch3D & dependensi pendukung
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c pytorch3d pytorch3d -y
pip install open3d opencv-python plyfile rich PyMCubes

# 4. Install CUDA Rasterizer SuGaR
cd gaussian_splatting/submodules/diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
cd ../../../

# 5. (Opsional tapi direkomendasikan) Install Nvdiffrast untuk export tekstur kilat
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
cd ..
```

---

### OPSI B: Native Windows 11 (Tanpa WSL)
Jika Anda memilih menjalankan langsung di Windows 11:
1. Pastikan terpasang **Visual Studio 2022 Community** dengan centang **Desktop development with C++**.
2. Pastikan terpasang **CUDA Toolkit 11.8**.
3. Di Anaconda Prompt Windows:
   ```cmd
   conda create -n sugar python=3.9 -y
   conda activate sugar
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
   pip install open3d opencv-python plyfile rich
   pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
   ```
4. Kompilasi submodule:
   ```cmd
   cd gaussian_splatting\submodules\diff-gaussian-rasterization
   pip install -e .
   cd ..\simple-knn
   pip install -e .
   cd ..\..\..
   ```

---

## 3. Alur Penggunaan Pipeline untuk Skripsi

### Langkah 1: Tempatkan Data Foto / Video
Sesuai SOP Lapangan, letakkan foto patung ke dalam folder:
* Contoh: `datasets/Dewa/model_01/images/` (150–300 foto)
* Jika ada video cadangan: `datasets/Dewa/model_01/raw_video/`
* Jika ada point cloud LiDAR acuan: `datasets/Dewa/model_01/ground_truth/`

### Langkah 2: Ekstraksi Video Cadangan (Jika Foto Langsung Kurang)
```bash
python pipeline_tools/01_extract_frames.py -v datasets/Dewa/model_01/raw_video/DJI_001.MP4 -o datasets/Dewa/model_01/images --fps 2.0 --statue_class Dewa --model_id model_01
```

### Langkah 3: Menjalankan COLMAP (SfM)
```bash
python pipeline_tools/02_run_colmap.py -s datasets/Dewa/model_01 --statue_class Dewa --model_id model_01
```

### Langkah 4: Melatih Model Baseline Vanilla 3DGS
```bash
python pipeline_tools/03_train_3dgs.py -s datasets/Dewa/model_01 --statue_class Dewa --model_id model_01 --iterations 30000
```
*Hasil akan tersimpan rapi di: `outputs/3dgs/Dewa/model_01/`*

### Langkah 5: Melatih Model SuGaR (Cepat & Bebas Error Memori)
```bash
python pipeline_tools/04_train_sugar.py -s datasets/Dewa/model_01 -c outputs/3dgs/Dewa/model_01 --statue_class Dewa --model_id model_01 --refinement_time short
```
*Dengan parameter `--refinement_time short`, tahap refinement hanya berlangsung sekitar 8–10 menit (bukan 2–3 hari).*
*Hasil mesh OBJ dan tekstur tersimpan di: `outputs/sugar/Dewa/model_01/textured_mesh/`*

### Langkah 6: Menghitung Metrik Evaluasi untuk Skripsi (dengan Auto-ICP)
```bash
python pipeline_tools/05_evaluate_comparison.py --statue_class Dewa --model_id model_01 --method SuGaR --model_path outputs/sugar/Dewa/model_01/textured_mesh/model.obj --gt_lidar datasets/Dewa/model_01/ground_truth/lidar.ply
```
*Skrip ini otomatis menjalankan **Auto-ICP Registration** untuk merapatkan posisi model ke LiDAR sebelum menghitung Chamfer Distance, dan otomatis menyimpan model hasil selaras di `outputs/logs/evaluation_reports/Dewa_model_01_SuGaR_aligned.ply` (bisa dibuka di Blender untuk gambar visual Bab 4).*
*Jika Anda ingin menyetel faktor pengali skala secara manual, cukup tambahkan argumen `--scale_factor <nilai>`.*

---

## 4. Eksekusi Otomatis Semalam untuk Seluruh 16 Patung (Batch Runner)

Anda tidak perlu menunggu patung satu per satu di depan layar. Cukup jalankan perintah ini sebelum tidur:
```bash
python pipeline_tools/batch_run_all.py
```
Perintah ini akan:
1. Memindai seluruh kelas (`Dewa`, `Dewi`, `Mitologi`, dst.) di folder `datasets/`.
2. Menjalankan COLMAP $\rightarrow$ Vanilla 3DGS $\rightarrow$ SuGaR secara berurutan.
3. Mencatat seluruh performa komputasi, durasi, dan puncak VRAM ke file log.

---

## 5. Melihat Hasil Rekapitulasi Data Skripsi

Setiap kali proses selesai, tabel ringkasan metrik skripsi Anda otomatis diperbarui di:
* **Format CSV**: `outputs/logs/evaluation_reports/summary_comparison.csv` *(Buka di Excel untuk membuat diagram perbandingan)*
* **Format Markdown**: `outputs/logs/evaluation_reports/summary_comparison.md` *(Langsung copy-paste ke Bab 4 Hasil dan Pembahasan Skripsi)*

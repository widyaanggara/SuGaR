# Optimasi mesh dan tampilan SuGaR dari checkpoint lama

Perintah berikut mempertahankan baseline, memakai ulang checkpoint coarse,
mengekstrak ulang mesh, menjalankan refinement, lalu mengevaluasi hasil lama dan
baru pada kamera uji yang sama. Jalankan setelah kode terbaru disalin ke server.

```bash
cd /home/anggara/SuGaR
conda activate sugar
python pipeline_tools/optimize_sugar.py --statue_class Dewi --model_id model_01 --gpu 0
```

Dari dashboard: **Pengaturan eksperimen → Training → Optimasi SuGaR**, pilih
patung, atur parameter, lalu klik **Antrekan optimasi SuGaR**. Pekerjaan masuk
antrean yang sama dan memakai GPU ID dari pengaturan. Hasil dashboard disimpan di
`outputs/quality/sugar/<Kelas>/<Model>/g<G>_it<iter>_fg<fg>_bg<bg>_d<depth>/`,
sehingga setiap konfigurasi punya folder sendiri. Jalur ini tidak menjalankan COLMAP,
3DGS, atau optimasi coarse ulang. Run yang terputus dapat dilanjutkan dengan
perintah yang sama; checkpoint dipakai ulang hanya jika identitas tahap cocok.

## Training SuGaR biasa juga fokus patung

Training SuGaR dari dashboard (tombol SuGaR atau Pipeline lengkap) kini memakai fokus patung secara
default: `outputs/sugar/<Kelas>/<Model>/object_mesh/object.ply` adalah bagian patung dari mesh refined,
bukan crop kotak kamera. Pengaturan lain tetap dari panel Parameter SuGaR (budget mesh untuk patung,
sekitar patung, dan background masing-masing; kotak foreground/background bawaan SuGaR tetap; Poisson
depth 10 bawaan SuGaR; Gaussian/segitiga; refinement). Total segitiga bisa sampai 3x budget, sehingga
VRAM perlu diuji pada satu patung dulu. Proyeksi vertex
ke titik sampel dimatikan agar urutan segitiga patung terjaga. Hasil lama yang meshnya seluruh scene
dianggap tidak cocok sehingga dijalankan ulang mulai dari ekstraksi mesh; checkpoint coarse dipakai ulang
bila dataset, baseline 3DGS, regularisasi, dan resolusi sama. `--no-statue_focus` pada
`04_train_sugar.py` mengembalikan perilaku lama. Bila fokus kamera gagal dihitung, log memberi peringatan
dan mesh seluruh scene dipakai.

## Mesh patung dan background (tetap SuGaR murni)

Semua Gaussian tetap terikat ke mesh, seperti SuGaR asli. Tidak ada Gaussian 3DGS bebas.

- **Mesh fokus patung** (`--statue_focus`, default aktif): di dalam kotak fokus kamera,
  bidang tanah dicari dengan RANSAC yang dibatasi arah atas kamera, lalu komponen 3D
  terhubung yang paling dekat dengan titik fokus kamera dipilih sebagai patung. Hanya
  titik patung yang masuk Poisson patung (`--statue_poisson_depth`, default 11), dan mesh
  dipangkas dengan aturan yang sama (`mesh/statue_region.npz`). Ini segmentasi geometris,
  bukan semantik: objek yang menempel pada patung tetap ikut. Jika tanah tidak terdeteksi,
  log memberi peringatan dan lantai yang menyatu dengan patung dapat ikut ke mesh.
- **`object_mesh/object.ply` = bagian patung dari mesh refined itu sendiri**, bukan hasil
  crop. Segitiga patung berada di urutan pertama mesh gabungan (`mesh/statue_split.json`).
- **Render tetap seluruh scene.** Tanah dan objek tetangga yang dibuang dari mesh patung
  mendapat mesh Poisson sendiri dengan resolusi foreground (bukan digabung ke background yang
  kasar), lalu digabung setelah patung: urutan mesh = patung, sekitar patung, background. Semua
  tetap dirender oleh Gaussian yang terikat mesh.
- **Blur background** dikurangi hanya lewat kapasitas SuGaR sendiri: Poisson background
  lebih rapat (`--poisson_depth`, default 11), 2.000.000 segitiga background, dan 3
  Gaussian per segitiga. Batasan bawaan SuGaR tetap ada (background jauh dari sedikit
  kamera tetap lebih halus dibanding metode dengan Gaussian bebas).

## Perubahan

- Batas fokus dihitung dari kamera dan digunakan sebagai batas foreground saat
  ekstraksi Poisson, **sebelum** penyederhanaan. Foreground dan background tetap
  direkonstruksi terpisah. Foreground kini **tanpa decimation** secara default;
  background dibatasi 1.000.000 segitiga. Mesh gabungan inilah yang dipakai
  refinement, bukan sekadar disimpan sebagai keluaran tambahan.
- Proyeksi vertex ke satu titik sampel terdekat dinonaktifkan pada jalur optimasi
  agar vertex mesh detail tidak dipaksa bertumpuk pada sampel yang sama.
  Pembersihan duplikat/segitiga degenerat tetap berjalan; ini bukan jaminan
  bentuk Poisson bebas noise.
- Background tetap masuk model untuk evaluasi seluruh gambar. Batas otomatis
  bukan segmentasi semantik; lantai dekat patung masih dapat ikut.
- Mesh foreground sebelum decimation disimpan untuk memeriksa kehilangan detail.
- Refinement memakai 3 Gaussian per segitiga dan **15.000 iterasi** (default
  upstream SuGaR), tidak lagi mewarisi iterasi run sumber (baseline biasanya 2.000).
  Resolusi, split, dan regularisasi tetap diwarisi dari run sumber. Poisson depth
  tetap 10 secara default. Iterasi sumber dicatat sebagai `source_refinement_iterations`.
- Proses training dan evaluasi dijalankan terpisah agar memori GPU training
  dilepas sebelum evaluasi. Penggunaan memori dan waktu refinement dapat meningkat.
- Hasil dan log optimasi tidak mengganti checkpoint atau rekap penelitian baseline.
  Render dan metrics.json evaluasi baseline diperbarui menggunakan protokol yang sama.

## Hasil

Folder default: `outputs/quality/sugar/Dewi/model_01/`.

| File | Isi |
|---|---|
| `object_mesh/object.ply` | Geometri refined yang dibatasi ke area fokus, tanpa tekstur |
| `surface/refined_surface.ply` | Geometri refined seluruh scene |
| `mesh/foreground_level0.3_predecimation.ply` | Foreground Poisson sebelum decimation/proyeksi |
| `evaluation/renders/` | Render refined seluruh gambar |
| `evaluation/metrics.json` | PSNR, SSIM, LPIPS dan skor setiap kamera |
| `quality_comparison.json` | Selisih skor terhadap refined lama; kamera diurutkan dari penurunan PSNR terbesar |
| `optimization_plan.json` | Parameter, sumber checkpoint, identitas dataset, dan batas fokus |
| `refinement_geometry.json` | Jumlah vertex, segitiga, dan Gaussian yang masuk refinement |

Delta PSNR/SSIM positif dan LPIPS negatif menunjukkan perbaikan. Laporan tetap
menampilkan penurunan bila hasil baru memburuk; tidak memilih checkpoint terbaik
berdasarkan kamera uji dan tidak mengganti render dengan model metode lain.

Untuk memeriksa rencana tanpa training, tambahkan `--plan_only`. Untuk perubahan
parameter berikutnya gunakan `--output_dir` berbeda. Opsi CLI tambahan:
`--gaussians_per_triangle {1,3,4,6}`, `--poisson_depth {6,...,12}`, dan
`--refinement_iterations {2000,7000,15000}` (default 15000).
`--foreground_triangles 0` mempertahankan resolusi foreground hasil Poisson;
nilai positif memberi batas segitiga. `--background_triangles 1000000` mengatur
background secara terpisah (0 juga berarti tanpa decimation).
Jika folder optimasi versi lama sudah berisi rencana, gunakan folder baru,
misalnya `--output_dir outputs/quality_detail/sugar/Dewi/model_01`.
Mesh tanpa decimation dengan 3 Gaussian/segitiga dapat melampaui VRAM 24 GB;
tidak ada klaim konfigurasi ini pasti muat. Tidak ada pengurangan kualitas
diam-diam ketika memori tidak cukup.

Ini adalah jalur optimasi yang perlu divalidasi di GPU, bukan jaminan PSNR naik
atau seluruh detail geometri pulih. Detail yang sudah hilang pada coarse tidak
otomatis kembali. Kualitas permukaan perlu dibandingkan pada bagian patung yang
sama; jumlah segitiga atau ukuran file bukan ukuran akurasi.

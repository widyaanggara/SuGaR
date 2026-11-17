# Optimasi mesh dan tampilan SuGaR dari checkpoint lama

Perintah berikut mempertahankan baseline, memakai ulang checkpoint coarse,
mengekstrak ulang mesh, menjalankan refinement, lalu mengevaluasi hasil lama dan
baru pada kamera uji yang sama. Jalankan setelah kode terbaru disalin ke server.

```bash
cd /home/anggara/SuGaR
conda activate sugar
python pipeline_tools/optimize_sugar.py --statue_class Dewi --model_id model_01 --gpu 0
```

Tidak ada tombol dashboard tambahan. Jalur ini tidak menjalankan COLMAP,
3DGS, atau optimasi coarse ulang. Run yang terputus dapat dilanjutkan dengan
perintah yang sama; checkpoint dipakai ulang hanya jika identitas tahap cocok.

## Perubahan

- Batas fokus dihitung dari kamera dan digunakan sebagai batas foreground saat
  ekstraksi Poisson, **sebelum** penyederhanaan. Foreground dan background tetap
  direkonstruksi terpisah, masing-masing dengan target decimation run sumber.
- Background tetap masuk model untuk evaluasi seluruh gambar. Batas otomatis
  bukan segmentasi semantik; lantai dekat patung masih dapat ikut.
- Mesh foreground sebelum decimation disimpan untuk memeriksa kehilangan detail.
- Refinement memakai 3 Gaussian per segitiga; jumlah iterasi, resolusi, split,
  dan regularisasi diwarisi dari run sumber. Poisson depth tetap 10 secara default.
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

Delta PSNR/SSIM positif dan LPIPS negatif menunjukkan perbaikan. Laporan tetap
menampilkan penurunan bila hasil baru memburuk; tidak memilih checkpoint terbaik
berdasarkan kamera uji dan tidak mengganti render dengan model metode lain.

Untuk memeriksa rencana tanpa training, tambahkan `--plan_only`. Untuk perubahan
parameter berikutnya gunakan `--output_dir` berbeda. Opsi CLI tambahan:
`--gaussians_per_triangle {1,3,4,6}` dan `--poisson_depth {6,...,12}`.

Ini adalah jalur optimasi yang perlu divalidasi di GPU, bukan jaminan PSNR naik
atau seluruh detail geometri pulih. Detail yang sudah hilang pada coarse tidak
otomatis kembali. Kualitas permukaan perlu dibandingkan pada bagian patung yang
sama; jumlah segitiga atau ukuran file bukan ukuran akurasi.

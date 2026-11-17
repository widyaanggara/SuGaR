# SuGaR: pengujian kualitas, waktu, dan mesh patung

Pipeline Lengkap tetap menjalankan COLMAP → 3DGS → SuGaR → training eksternal →
evaluasi dan mesh eksternal. Pengaturan tambahan disalin ke setiap pekerjaan saat
masuk antrean. Mengubah pengaturan berikutnya tidak mengubah pekerjaan yang sudah
menunggu. Resolusi, split setiap delapan foto, dan evaluasi refined seluruh gambar
tetap mengikuti protokol bersama.

## Penggunaan di dashboard

1. Uji satu patung terlebih dahulu. Pilih preset **Uji kualitas/waktu · 3 Gaussian
   + medium** untuk membandingkan kapasitas/iterasi, atau **Uji latar kompleks ·
   sdf + long** untuk membandingkan regularisasi. Keduanya eksperimental pada
   dataset penelitian ini, bukan konfigurasi yang terbukti menaikkan PSNR atau
   pasti lebih cepat. Simpan hasil pembanding sebelum retraining. Lebih banyak
   Gaussian juga bisa meningkatkan waktu dan VRAM.
2. Pada Training pilih **Geometri PLY · tanpa baking tekstur** bila belum perlu OBJ
   bertekstur. Mode ini menjadi pilihan awal untuk browser baru. Evaluasi tetap
   merender checkpoint Gaussian refined yang sama; file PLY berisi permukaan
   refined dari `_points` dan `_surface_mesh_faces`, bukan Gaussian point cloud.
   Mode OBJ tetap tersedia. Pekerjaan lama/API/CLI tetap default OBJ.
3. Aktifkan **Render coarse dan refined** untuk diagnosis blur pada Pipeline
   Lengkap. `evaluation_coarse/` berisi render dan metrics.json terpisah. Preview
   dashboard menampilkan `sugar_coarse` di samping hasil refined. Ini menambah
   waktu evaluasi, bukan memperbaiki rendering dengan sendirinya. Log diagnostik
   tidak ditulis ke rekap metode sehingga skor/waktu/VRAM refined tidak terganti.
4. Fokus mesh kini **otomatis dan hanya untuk SuGaR**. Tidak ada input batas
   pada kartu model. Pipeline membaca posisi dan rotasi kamera dari checkpoint
   3DGS, mencari pusat perpotongan arah pandang secara robust, lalu membatasi
   mesh dengan kotak berdasarkan jarak, bidang pandang dan margin 15%.
   Pengaturan manual lama tidak dipakai oleh pekerjaan dashboard baru.
5. Jalankan Pipeline Lengkap seperti biasa. Pemfokusan diterapkan otomatis pada
   mesh hasil akhir di `outputs/sugar/Kelas/model_id/object_mesh/object.ply`, tanpa
   tombol atau tautan tambahan pada kartu model.
   Hasil ablasi berada di struktur yang sama di `outputs/ablations/scaffold_anchor/`.

Metode ini membatasi luas scene, bukan segmentasi semantik. Ia mengasumsikan
kamera diarahkan ke patung utama. Lantai atau objek dekat bisa tetap ikut;
patung yang tidak berada di pusat bidikan bisa terpotong. Mesh asli dipertahankan.
Kamera terlalu sejajar, arah pandang tidak konsisten, atau crop kosong menghasilkan
peringatan dan error.json; training metode berikutnya tetap dilanjutkan.
Fingerprint mesh dan cameras.json mencegah pemakaian ulang crop yang kedaluwarsa.
2DGS dan RaDe-GS tidak menjalankan ekspor crop tambahan ini.

Ekspor geometri saja dapat memanfaatkan coarse/refinement yang cocok melalui
pemulihan tahap. Ketika seluruh training dipakai kembali, sesi ekspor tidak
menggantikan catatan waktu/VRAM training pada rekap. Waktu tiap tahap tersedia
di log training JSON dan `pipeline_result.json:stage_timings_seconds`.

## CLI

```bash
python pipeline_tools/batch_run_all.py --class_name Mitologi --model_id model_02 \
  --skip_colmap --include_external \
  --python_2dgs /path/to/2dgs/bin/python \
  --python_radegs /path/to/radegs/bin/python \
  --sugar_mesh_export surface --sugar_diagnostics
```

CLI batch dan SuGaR juga memakai fokus otomatis. Untuk menonaktifkan ekspor
fokus pada CLI gunakan `--object_region_json null`. Hasil crop adalah geometri
PLY tanpa tekstur UV; evaluasi visual tetap memakai render seluruh scene.
Evaluasi geometri utama tetap memakai sumber scene semula sehingga perubahan
crop tidak diam-diam mengganti protokol penelitian.

Menaikkan PSNR/SSIM atau menurunkan LPIPS dan waktu secara bersamaan masih perlu
dibuktikan lewat percobaan GPU. Integrasi parameter dan pemeriksaan CPU tidak
membuktikan peningkatan kualitas visual atau batas VRAM pada server.

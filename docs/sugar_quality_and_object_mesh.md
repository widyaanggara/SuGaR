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
4. Sesudah COLMAP tersedia, klik **Atur batas patung** pada kartu model dan isi
   minimum/maksimum XYZ berdasarkan mesh asli yang masih memakai koordinat
   COLMAP. Jangan memakai koordinat LiDAR atau mesh yang telah digeser, diputar,
   dinormalisasi atau diskalakan di viewer. Dalam viewer mesh, gunakan kotak
   seleksi/crop untuk melihat koordinat batas objek. Sertakan mahkota, tangan,
   atribut dan alas yang memang menjadi bagian patung. Tidak ada angka batas
   universal yang aman untuk semua patung.
5. Simpan batas per patung sekali, kemudian antrekan **Pipeline Lengkap** untuk
   semua target. Dengan COLMAP yang sudah benar, gunakan mulai dari hasil COLMAP
   agar sistem koordinat yang dipakai batas objek tetap sama. Jika SfM dihitung
   ulang, simpan ulang batasnya: fingerprint COLMAP mencegah crop dengan batas lama.

Mesh objek baseline tersedia melalui tautan pada kartu model ketika ekspor valid.
File setiap metode adalah:

```
outputs/{sugar,2dgs,radegs}/Kelas/model_id/object_mesh/object.ply
outputs/{sugar,2dgs,radegs}/Kelas/model_id/object_mesh/region.json
```

Ablasi memakai struktur yang sama di `outputs/ablations/scaffold_anchor/`.
Ini salinan geometri PLY tanpa tekstur UV. Mesh asli, checkpoint refined, tekstur
dan evaluasi seluruh gambar tetap tersimpan. Pembatasan menggunakan kotak XYZ;
objek lain/lantai yang masih berada di dalam kotak tidak otomatis tersegmentasi.
Pemotongan hanya mempertahankan segitiga dalam kotak dan dapat memotong bagian
patung bila kotaknya terlalu sempit. Ambang fragmen 50 segitiga dapat diubah ke 0
untuk mempertahankan komponen kecil. Komponen terbesar tidak dipilih otomatis
karena mahkota atau aksesori patung dapat terpisah.

Jika batas belum diisi, mesh objek tidak dibuat. Jika crop kosong, COLMAP berubah,
atau ekspor objek gagal, ada `[WARNING][OBJECT]`, file `object_mesh/error.json`,
dan indikator periksa batas/log pada kartu model; pekerjaan metode lain tetap
dilanjutkan. Mesh objek lama tidak dianggap hasil baru yang sukses. Ini tidak
menyembunyikan kegagalan training atau ekstraksi mesh utama: kegagalan utama
tetap membuat pekerjaan gagal.

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

CLI membaca `datasets/Kelas/model_id/object_region.json` jika ada. Flag
`--object_region_json` dapat memasukkan snapshot JSON; `null` menonaktifkan crop
untuk pekerjaan itu. Dashboard memasukkan snapshot secara eksplisit.

Untuk membersihkan mesh yang sudah ada, training tidak perlu diulang:

```bash
python pipeline_tools/object_mesh.py --statue_class Mitologi --model_id model_02 --method sugar
python pipeline_tools/object_mesh.py --statue_class Mitologi --model_id model_02 --method radegs
```

Gunakan environment yang memiliki Open3D. CLI crop langsung gagal dengan pesan
jelas jika batas/hasil tidak valid; pipeline otomatis menggunakan mode opsional.
Evaluasi geometri utama masih memakai sumber geometri scene yang tercatat.
Jika mesh crop dipakai untuk penelitian, terapkan wilayah objek yang sama pada
semua metode dan laporkan protokol crop tersebut; skor visual seluruh gambar
tidak boleh diklaim meningkat hanya karena bagian background dibuang.

Menaikkan PSNR/SSIM atau menurunkan LPIPS dan waktu secara bersamaan masih perlu
dibuktikan lewat percobaan GPU. Integrasi parameter dan pemeriksaan CPU tidak
membuktikan peningkatan kualitas visual atau batas VRAM pada server.

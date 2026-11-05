# Pemulihan proses yang terputus

Klik **Antrekan Ulang** pada pekerjaan `INTERRUPTED`, `FAILED`, atau `CANCELLED`.

- Sistem memakai pengaturan asli dari server, mengaktifkan pemulihan, dan memberi prioritas setelah pekerjaan aktif selesai. Pekerjaan aktif tidak dihentikan.
- Kalau model itu sudah menunggu dengan konfigurasi yang sama, antreannya diprioritaskan tanpa membuat duplikat. Konfigurasi berbeda harus dibatalkan dulu, agar tidak diganti diam-diam.
- Kalau antrean dijeda, pemulihan tetap menunggu tombol **Lanjutkan Antrean**. Setelah restart server, periksa GPU/proses lama sebelum melanjutkan.
- Urutan tabel mengikuti urutan eksekusi: pekerjaan aktif, pemulihan yang menunggu, lalu antrean biasa. Setelah restart tanpa pekerjaan aktif, hasil **Antrekan Ulang** tampil paling atas dan tetap dijeda sampai **Lanjutkan Antrean** ditekan. Prioritas tersimpan dan bertahan jika server dimulai ulang lagi.

## Apa yang dapat dilanjutkan

**3DGS:** snapshot training baru disimpan setiap 1.000 iterasi, juga pada milestone ekspor. Snapshot memuat parameter Gaussian, state Adam, statistik densifikasi, iterasi, RNG Python/NumPy/PyTorch/CUDA, dan urutan/sisa kamera. Checkpoint PLY saja bukan checkpoint optimizer. Checkpoint milestone lama tetap diekspor untuk kompatibilitas. Iterasi 7.000 dan 30.000 tetap menjadi hasil yang digunakan pipeline.

**SuGaR:** coarse (density, DN consistency, atau SDF) dan refinement menyimpan snapshot setiap 500 iterasi dan di akhir. State mencakup parameter model, optimizer, densifier/neighbor buffers, RNG, urutan gambar dan posisi di epoch, tingkat SH, serta loss/iterasi. Tahap coarse, ekstraksi mesh, refinement, dan ekspor tekstur mempunyai catatan selesai terpisah. Hasil tahap yang lengkap dan masih cocok dipakai kembali; tahap ekstraksi mesh/tekstur yang terputus di tengah diulang dari awal tahap tersebut.

**COLMAP:** pipeline otomatis dengan pemulihan aktif langsung memakai `images/` dan `sparse/0` yang lengkap, termasuk hasil impor teman tanpa catatan pengaturan. Tidak mengulang COLMAP karena pengaturan panel berbeda, sehingga checkpoint training tetap mengacu pada rekonstruksi yang sama. Jika hasil belum lengkap, database fitur/matching pada konfigurasi yang sama dapat digunakan kembali oleh COLMAP. Mapper/undistorsi yang belum selesai dapat perlu dijalankan ulang; implementasi ini tidak menyimpan state optimizer internal mapper. Opsi **Hanya model dengan hasil COLMAP lengkap** melarang COLMAP dijalankan ulang jika berkas hasil hilang. Gunakan **COLMAP saja** bila memang ingin menghitung SfM ulang.

**Evaluasi:** evaluasi yang sudah lengkap dan cocok dilewati oleh pipeline. Evaluasi yang terputus masih dijalankan ulang per metode; belum ada pemulihan per gambar evaluasi.

## Pemeriksaan dan penyimpanan

Snapshot baru berada di `outputs/<3dgs|sugar>/<kelas>/<model>/recovery/`. Dua slot berputar disimpan per tahap/konfigurasi: checkpoint terbaru dan cadangan sebelumnya. Penulisan menggunakan file sementara, flush/fsync, lalu replace; snapshot lama tetap tersedia bila penulisan gagal. Metadata bukan satu-satunya pemeriksaan: worker memuat dan memeriksa envelope, fingerprint, model dan optimizer sebelum melanjutkan. Checkpoint terbaru yang rusak dicoba dengan cadangan; bila seluruh kandidat gagal, proses berhenti dengan pesan, tanpa diam-diam mulai dari nol.

Identitas pemulihan mencakup signature dataset/COLMAP, resolusi, parameter training, baseline/mesh sumber, versi runtime yang relevan, dan hash berkas implementasi yang dicatat. Memulai training baru dengan **Lewati tahap yang selesai** nonaktif / `--no_resume` menghasilkan generasi baru, agar snapshot run lama tidak tercampur. Riwayat generasi lama dipertahankan. Perubahan input/pengaturan/kode dapat membuat snapshot lama tidak sesuai.

Catatan selesai SuGaR mencatat signature output dan input tiap tahap. Ekspor OBJ juga memeriksa berkas MTL dan tekstur yang dirujuk. Lihat log `[RESUME]` untuk checkpoint/iterasi yang benar-benar dipakai dan `[CHECKPOINT]` untuk snapshot yang baru disimpan.

## Batas penting untuk pekerjaan lama

Fitur ini berlaku penuh untuk run yang dibuat setelah pembaruan. Checkpoint lama yang hanya memuat model, atau tidak mempunyai metadata dataset/pengaturan/RNG yang cukup, **tidak otomatis dianggap sebagai snapshot pemulihan yang aman**. Tahap lengkap yang dikenali pipeline masih dapat dipakai; tahap parsial tanpa snapshot yang cocok dapat dimulai ulang dan alasannya ditulis di log. Iterasi yang belum sempat disimpan tidak dapat dikembalikan.

Siapkan ruang disk untuk dua snapshot optimizer per tahap, di samping hasil model. Snapshot besar menambah waktu I/O pada batas penyimpanan; tidak ditulis setiap iterasi. Jangan menghapus folder recovery pada run yang masih ingin dipulihkan.

Durasi dan peak VRAM pada log sesi/CSV tetap mengacu pada proses sesi tersebut, termasuk pemuatan checkpoint, bukan akumulasi seluruh percobaan sebelum terputus. Untuk perbandingan biaya komputasi skripsi, simpan histori antrean/log percobaan dan bedakan run penuh dengan run pemulihan. Tahap yang dipakai kembali tidak mengulang seluruh durasinya pada sesi baru.

Pengujian lokal menggunakan file sementara, model/optimizer NumPy kecil, serta proses dummy. Tes memeriksa kontinuitas optimizer/RNG/kursor kamera, fallback checkpoint rusak, penulisan gagal, pemisahan konfigurasi/generasi, cache tahap SuGaR, API retry, prioritas, dan urutan eksekusi tanpa preemption. Pemulihan CUDA penuh tetap perlu diuji pada environment GPU server; tidak ada training GPU yang dijalankan di laptop. Kelanjutan state bukan janji hasil CUDA identik bit demi bit, karena kernel training dapat nondeterministik.

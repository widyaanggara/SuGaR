# Ekspor hasil evaluasi ke Excel

Tombol **Ekspor Excel · Semua hasil** mengunduh `evaluasi_baseline_dan_ablasi.xlsx`.
Ekspor menyertakan seluruh baseline dan studi ablasi, terlepas dari tab, metode,
atau ambang geometri yang sedang dipilih. File dibuat di memori; tidak perlu
memasang paket Excel tambahan pada server.

- **Perbandingan:** selisih ablasi dikurangi baseline, hanya untuk pasangan
  patung/metode yang memiliki kedua hasil. Satu baris per patung, 3DGS dan SuGaR
  berdampingan. Selisih bukan ukuran signifikansi statistik.
- **Baseline** dan **Studi ablasi:** hasil terakhir berstatus `COMPLETED`,
  digabung dengan aturan yang sama seperti dashboard. Satu baris per patung;
  metrik visual, geometri pada ambang 2 dan 5 cm, peak VRAM, waktu tahap, dan
  jumlah gambar dikelompokkan per metode.
- **Panduan:** sumber data, arti metrik, data yang belum tersedia, dan batasan
  perbandingan waktu maupun kualitas.

Header dan identitas patung dibekukan saat scrolling. Kolom dapat difilter,
baris diberi warna selang-seling, angka disimpan sebagai angka, dan teks nama
patung selalu disimpan sebagai teks literal. Nilai `—` berarti belum tersedia,
bukan nol. Tabel lebar sengaja mempertahankan kedua metode berdampingan.

Jumlah gambar berasal dari `evaluation/metrics.json` yang cocok dengan nilai
metrik visual pada laporan. Evaluasi baru menyimpan `registered_images`,
`train_images`, dan `test_images`. Untuk hasil lama, jumlah test dapat diambil
dari `per_view`; registered dan train tidak ditebak dari isi folder dataset.
Jalankan evaluasi visual kembali untuk mencatat jumlah yang belum tersedia;
training ulang tidak diperlukan jika checkpoint dan konfigurasi masih cocok.

Waktu/VRAM berasal dari baris training, bukan baris evaluasi. Waktu SuGaR tidak
mencakup tahap 3DGS; waktu ablasi belum mencakup Scaffold. Peak VRAM antar tahap
tidak boleh dijumlahkan. Ringkasan mengikuti penggabungan hasil terakhir pada
dashboard, bukan arsip per run/seed; pastikan konfigurasi tetap sebanding sebelum
menarik kesimpulan lintas run.

Endpoint CSV lama tetap tersedia untuk kompatibilitas. UI menggunakan
`GET /api/metrics.xlsx`. Restart dashboard dan muat ulang halaman sesudah
memperbarui kode.

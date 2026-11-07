# Ekspor hasil evaluasi ke Excel

Tombol **Ekspor Excel · Semua hasil** mengunduh `evaluasi_baseline_dan_ablasi.xlsx`.
Ekspor menyertakan seluruh baseline dan studi ablasi, terlepas dari tab, metode,
atau ambang geometri yang sedang dipilih. File dibuat di memori; tidak perlu
memasang paket Excel tambahan pada server.

- **Perbandingan:** selisih ablasi dikurangi baseline, hanya untuk pasangan
  patung/metode yang memiliki kedua hasil. Satu baris per metode: 3DGS lalu SuGaR di bawahnya untuk patung yang sama. Selisih bukan ukuran signifikansi statistik.
- **Baseline** dan **Studi ablasi:** hasil terakhir berstatus `COMPLETED`,
  digabung dengan aturan yang sama seperti dashboard. Dua baris per patung jika kedua metode tersedia, 3DGS lalu SuGaR;
  metrik visual, geometri pada ambang 2 dan 5 cm, peak VRAM, waktu tahap, dan
  jumlah gambar dikelompokkan per metode.
- **Panduan:** sumber data, arti metrik, data yang belum tersedia, dan batasan
  perbandingan waktu maupun kualitas.

Header dan identitas patung dibekukan saat scrolling. Kolom dapat difilter,
pasangan baris satu patung diberi warna yang sama, bergantian antarpatung, angka disimpan sebagai angka, dan teks nama
patung selalu disimpan sebagai teks literal. Nilai `—` berarti belum tersedia,
bukan nol. Kolom metode membedakan hasil 3DGS dan SuGaR tanpa menggandakan kolom metrik.

Jumlah gambar berasal dari `evaluation/metrics.json` yang cocok dengan nilai
metrik visual pada laporan. Evaluasi baru menyimpan `registered_images`,
`train_images`, dan `test_images`. Untuk hasil lama, ekspor membaca
`outputs/3dgs/<kelas>/<model>/cameras.json` (atau folder 3DGS pada eksperimen
ablasi). Evaluasi 3DGS dan SuGaR sama-sama memakai daftar kamera tersebut.
Nama gambar dideduplikasi dan diurutkan seperti loader evaluasi. Daftar gambar
test hasil pembagian menurut `eval_split_interval` harus cocok persis dengan
nama pada `per_view`. Registered adalah jumlah kamera unik; train dihitung
sebagai registered dikurangi test. Tidak perlu training atau evaluasi ulang.

Jika file kamera hilang, rusak, atau split tidak cocok, registered/train tetap
`—` dan jumlah test tetap diambil dari `per_view`. Ekspor tidak menebak dari
jumlah foto mentah atau model COLMAP pada dataset yang mungkin sudah berubah.

Waktu/VRAM berasal dari baris training, bukan baris evaluasi. Waktu SuGaR tidak
mencakup tahap 3DGS; waktu ablasi belum mencakup Scaffold. Peak VRAM antar tahap
tidak boleh dijumlahkan. Ringkasan mengikuti penggabungan hasil terakhir pada
dashboard, bukan arsip per run/seed; pastikan konfigurasi tetap sebanding sebelum
menarik kesimpulan lintas run.

Endpoint CSV lama tetap tersedia untuk kompatibilitas. UI menggunakan
`GET /api/metrics.xlsx`. Restart dashboard dan muat ulang halaman sesudah
memperbarui kode.

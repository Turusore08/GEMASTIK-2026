# PANDUAN IMPLEMENTASI BASELINE 1: LFCC + LCNN (OFFICIAL GITHUB CLONE)
**Arsitektur:** Linear Frequency Cepstral Coefficients (LFCC) + Light CNN (LCNN)
**Peran:** Sebagai batas bawah performa (*lower-bound baseline*) menggunakan arsitektur orisinal dari literatur (Wu et al., 2018).

---

## INSTRUKSI UNTUK AI AGENT (PROMPT JUPYTER NOTEBOOK GENERATOR)

*(Salin teks di bawah ini dan berikan kepada AI Agent seperti GitHub Copilot, ChatGPT, atau Claude untuk membuatkan file `.ipynb`)*

**Prompt AI Agent:**
> "Kamu adalah seorang Senior AI Engineer yang fokus pada riset Audio Anti-Spoofing (ASVspoof). Saya sedang membangun Baseline 1 untuk eksperimen saya menggunakan arsitektur LFCC + LCNN.
> 
> **Konteks:** Saya sudah melakukan *clone* pada repositori GitHub resmi Light CNN (berdasarkan paper Wu et al., 2018). Repositori tersebut berada di direktori lokal saya (D:\Lomba\Gemastik 2026\Data Mining\panduan\github). 
> 
> **Tugas:** Buatkan saya skrip lengkap dalam bentuk **Jupyter Notebook (.ipynb)** yang siap dijalankan (reproducible). Skrip ini harus memuat tahapan dari ekstraksi fitur hingga evaluasi metrik akhir.
> 
> **Syarat dan Spesifikasi Kode:**
> 1. **Reproduksibilitas:** Awali *notebook* dengan pengaturan `random seed` untuk PyTorch, NumPy, dan Python agar hasilnya deterministik.
> 2. **Ekstraksi Fitur (Front-End):** Buat fungsi ekstraksi LFCC menggunakan pustaka `torchaudio` atau `spafe`. Input berupa file `.wav` (16kHz), dan output berupa tensor 2D. Tambahkan fungsi *padding/truncating* agar dimensi waktu (time frames) seragam untuk setiap *batch*.
> 3. **Integrasi Model LCNN (Back-End):** >    - Berikan kode untuk mengimpor arsitektur LCNN dari folder hasil *clone* (misalnya `from light_cnn import LightCNN`). 
>    - Karena LCNN asli sering kali dikembangkan untuk gambar wajah (input 3 *channels* RGB atau 1 *channel* grayscale ukuran tertentu), berikan kode *wrapper* atau modifikasi lapisan `Conv2d` pertamanya agar sesuai dengan dimensi tensor LFCC (1 *channel*).
>    - Ubah atau pastikan *classifier head* (lapisan terakhir) menghasilkan 2 kelas (*Bona fide* vs *Spoof*).
> 4. **Metrik Evaluasi (Matrix):** Ini adalah bagian paling krusial. Pada fase validasi/evaluasi, hitung dan tampilkan metrik berikut:
>    - **Equal Error Rate (EER):** Buatkan fungsi untuk menghitung EER menggunakan modul `scipy.optimize.brentq` dan `sklearn.metrics.roc_curve`.
>    - **tandem Detection Cost Function (t-DCF):** Berikan fungsi atau *wrapper* untuk menghitung nilai minimum t-DCF (min t-DCF) sesuai protokol standar ASVspoof.
> 5. **Training & Validation Loop:** Buat *loop* pelatihan menggunakan `Adam optimizer` dan `CrossEntropyLoss`. Di setiap akhir *epoch*, cetak (print) *Loss*, *Accuracy*, *EER*, dan *min t-DCF* dalam format yang rapi (seperti matriks log).
> 6. **Pengujian Lintas-Dataset (Cross-Dataset Evaluation ASVspoof 5 Lokal):**
>    - Tambahkan sel khusus untuk mengevaluasi model pada dataset **ASVspoof 5** secara **lokal** sesuai dengan panduan **data.md**.
>    - Terapkan **Stratified Quota Sampling** menggunakan pandas dari file `.tsv` lokal:
>      1. Buat fungsi `get_sampled_dataframe(tsv_path, max_bonafide, samples_per_attack)` untuk membaca berkas `.tsv` lokal (`ASVspoof5.dev.track_1.tsv`) dengan parameter `sep=' '` dan kolom-kolom: `SPEAKER_ID`, `FLAC_FILE_NAME`, `SPEAKER_GENDER`, `CODEC`, `CODEC_Q`, `CODEC_SEED`, `ATTACK_TAG`, `ATTACK_LABEL`, `KEY`, `TMP`.
>      2. Filter kelas `bonafide` sebanyak `max_bonafide` (misal 2500) dan kelas `spoof` dengan *groupby* `ATTACK_LABEL` lalu ambil `samples_per_attack` (misal 250) per grup jenis serangan agar seimbang.
>      3. Buat kelas `ASVspoof5Dataset(Dataset)` yang menerima DataFrame hasil sampling dan `audio_dir` (path root folder audio). Di dalam `__init__`, gunakan `os.walk(audio_dir)` untuk memindai seluruh subfolder secara rekursif (misalnya `flac_D_aa/flac_D/`, `flac_D_ab/flac_D/`, dll.) dan bangun map/dictionary nama file ke path absolutnya.
>      4. Di dalam `__getitem__`, ambil file audio `.flac` menggunakan path absolut dari map pemetaan tersebut, lakukan *padding/truncating* agar panjangnya tepat 64000 samples, ubah kolom `KEY` menjadi label biner (0 untuk `bonafide`, 1 untuk `spoof`), dan kembalikan `(waveform_tensor, label_tensor)`.
>      5. Gunakan `DataLoader` untuk memuat data pengujian ini dan jalankan evaluasi untuk menghitung EER dan min t-DCF.
> 
> Tolong berikan penjelasan Markdown di dalam Jupyter Notebook untuk setiap sel (cell) kodenya agar mudah saya laporkan ke dalam jurnal/makalah riset."
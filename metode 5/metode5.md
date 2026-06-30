# PANDUAN IMPLEMENTASI METODE 5: THE PROPOSED MODEL (FULL ARCHITECTURE)
**Arsitektur:** Fusi CQT-LFCC + AASIST Backbone + Temporal Attention Mechanism
**Peran:** Sebagai model utama (*Proposed Method*) yang diharapkan menghasilkan nilai EER dan min t-DCF terendah. Model ini membuktikan bahwa fusi fitur hibrida yang dipadukan dengan pemantauan dependensi waktu (*temporal attention*) adalah solusi terbaik untuk mendeteksi *deepfake* modern.

---

## INSTRUKSI UNTUK AI AGENT (PROMPT JUPYTER NOTEBOOK GENERATOR)

*(Salin teks di bawah ini dan berikan kepada AI Agent seperti GitHub Copilot, ChatGPT, atau Claude untuk membuatkan file `.ipynb`)*

**Prompt AI Agent:**
> "Kamu adalah seorang Lead AI Researcher di bidang Speech Anti-Spoofing. Saya sedang membangun **Proposed Model (Metode 5)** untuk makalah saya. Model ini akan menggunakan **Fusi CQT-LFCC** sebagai *front-end*, **AASIST** sebagai *backbone graph extractor*, dan **Temporal Attention Mechanism** sebagai *classifier head* (menggantikan *pooling/linear layer* standar).
> 
> **Konteks & Syarat Eksekusi:** Buat skrip **Jupyter Notebook (.ipynb) mandiri** yang *reproducible* dan siap dijalankan di Kaggle kernel tanpa *path* lokal.
> 
> **Tugas dan Spesifikasi Kode Penuh:**
> 1. **Persiapan Environment:**
>    - Jalankan `!git clone https://github.com/clovaai/aasist.git`.
>    - Tambahkan `sys.path.append('./aasist')`.
> 
> 2. **Front-End (Fusi Multi-Domain):**
>    - Buat fungsi ekstraksi untuk **CQT** dan **LFCC** menggunakan `torchaudio` atau `librosa`.
>    - Sama seperti pada *Ablation Study*, buat fungsi interpolasi (`torch.nn.functional.interpolate`) untuk menyamakan dimensi waktu (*time frames*) kedua fitur ini sebelum di-*concatenate* secara frekuensi.
> 
> 3. **Back-End (AASIST + Adapter):**
>    - Impor AASIST. Buang lapisan `Sinc-Conv` bawaan.
>    - Buat **Adapter Layer** (`Conv2D`) untuk mentransformasi dimensi tensor fusi CQT-LFCC agar sesuai dengan input *Graph Attention layer* pertama dari AASIST.
> 
> 4. **Inovasi Inti: Temporal Attention Mechanism:**
>    - Setelah pemrosesan graf oleh AASIST, biasanya ada operasi *readout/pooling*. **Modifikasi bagian ini!**
>    - Sebelum *pooling* akhir, masukkan tensor ke dalam sebuah kelas `TemporalAttention` kustom (berbasis *Query, Key, Value* atau menggunakan `torch.nn.MultiheadAttention`). 
>    - Tujuan layer ini adalah memfokuskan *weight* pada bingkai waktu (*time frames*) tertentu yang mengandung anomali artifak *deepfake* (seperti jeda bicara atau ritme napas buatan).
>    - Setelah melewati *Temporal Attention*, lakukan *flattening* dan lewati *Linear classifier* untuk *output* 2 kelas (*Bona fide* vs *Spoof*).
> 
> 5. **Strategi Dataset & Cross-Evaluation (ASVspoof 5 Lokal):**
>    - **Training & Validation (ASVspoof 5 Lokal):** Pemuatan dataset lokal ASVspoof 5 dengan file protokol `ASVspoof5.train.tsv` (untuk training) dan `ASVspoof5.dev.track_1.tsv` (untuk validation/dev) menggunakan teknik **Stratified Quota Sampling** dari `data.md` untuk menghemat memori.
>    - **Testing (Stress-Test - ASVspoof 5 Lokal):**
>      - Terapkan pemuatan dataset **ASVspoof 5 secara lokal** (contoh path: `/kaggle/input/datasets/raditya0/asvspoof5` dengan fallback ke `D:/Lomba/Gemastik 2026/Data Mining/Dataset_ASVspoof5_Sampled`) sesuai panduan **data.md**.
>      - Terapkan **Stratified Quota Sampling** menggunakan pandas dari file `.tsv` lokal:
>        1. Buat fungsi `get_sampled_dataframe(tsv_path, max_bonafide, samples_per_attack)` untuk membaca berkas `.tsv` (`ASVspoof5.dev.track_1.tsv`) menggunakan `pandas` dengan `sep=' '` dan kolom-kolom: `SPEAKER_ID`, `FLAC_FILE_NAME`, `SPEAKER_GENDER`, `CODEC`, `CODEC_Q`, `CODEC_SEED`, `ATTACK_TAG`, `ATTACK_LABEL`, `KEY`, `TMP`.
>        2. Filter kelas `bonafide` sebanyak `max_bonafide` (misal 2500) dan kelas `spoof` dengan *groupby* `ATTACK_LABEL` lalu ambil `samples_per_attack` per jenis serangan agar seimbang.
>        3. Buat kelas `ASVspoof5Dataset(Dataset)` yang menerima DataFrame hasil sampling dan `audio_dir` (path root folder audio). Di dalam `__init__`, gunakan `os.walk(audio_dir)` untuk memindai seluruh subfolder secara rekursif (misalnya `flac_D_aa/flac_D/`, `flac_D_ab/flac_D/`, dll.) dan bangun map/dictionary nama file ke path absolutnya.
>        4. Di dalam `__getitem__`, ambil file audio `.flac` menggunakan path absolut dari map pemetaan tersebut, lakukan *padding/truncating* agar panjangnya tepat 64600 samples (sesuai standar input AASIST), ubah kolom `KEY` menjadi label biner (0 untuk `bonafide`, 1 untuk `spoof`), dan kembalikan `(waveform_tensor, label_tensor)`.
>        5. Inisialisasi `DataLoader` untuk evaluasi lintas-dataset ini untuk menghitung EER dan min t-DCF.
> 
> 6. **Metrik & Training Loop:**
>    - Kalkulasi **EER (Equal Error Rate)** dan **min t-DCF**.
>    - Gunakan `AdamW` optimizer, `CosineAnnealingLR`, dan *Cross Entropy Loss*.
>    - Tampilkan ringkasan metrik (Loss, Akurasi, EER, t-DCF) di akhir setiap *epoch*.
> 
> Tuliskan kode yang elegan dan modular. Berikan blok *Markdown* yang sangat jelas sebelum kelas `TemporalAttention` untuk menjelaskan secara matematis bagaimana mekanisme atensi tersebut menangkap 'long-term sequential dependencies'."
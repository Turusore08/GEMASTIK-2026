# PANDUAN IMPLEMENTASI METODE 4: UJI ABLASI FUSI CQT-LFCC + AASIST
**Arsitektur:** Fusi Fitur CQT-LFCC (Front-End) + AASIST Backbone (Back-End)
**Peran:** Sebagai model *Ablation Study* (Uji Ablasi). Metode ini membuktikan bahwa fusi fitur hibrida usulan kita lebih superior dalam menangkap artifak *deepfake* dibandingkan *Sinc-filter* bawaan AASIST.

---

## INSTRUKSI UNTUK AI AGENT (PROMPT JUPYTER NOTEBOOK GENERATOR)

*(Salin teks di bawah ini dan berikan kepada AI Agent seperti GitHub Copilot, ChatGPT, atau Claude untuk membuatkan file `.ipynb`)*

**Prompt AI Agent:**
> "Kamu adalah seorang Senior AI Engineer spesialis Audio Forensics dan Graph Neural Networks. Saya sedang menyusun **Ablation Study (Metode 4)** untuk eksperimen saya. Saya akan mengganti *front-end* bawaan AASIST dengan **Fusi Fitur CQT dan LFCC**, sementara *backbone* graf-nya tetap menggunakan AASIST.
> 
> **Konteks & Syarat Eksekusi:** Skrip ini harus berupa **Jupyter Notebook (.ipynb) mandiri** yang dirancang untuk dijalankan di kernel Kaggle. Jangan gunakan referensi path lokal.
> 
> **Tugas dan Spesifikasi Kode:**
> 1. **Persiapan Environment:** >    - Berikan perintah `!git clone https://github.com/clovaai/aasist.git` di sel pertama.
>    - Gunakan `sys.path.append('./aasist')` untuk integrasi *module*.
> 
> 2. **Front-End (Modul Fusi CQT-LFCC):** >    - Buatkan *class* atau fungsi ekstraktor fitur menggunakan `librosa` atau `torchaudio`.
>    - Ekstrak **CQT** (Constant-Q Transform) dan **LFCC** (Linear Frequency Cepstral Coefficients) dari input audio `.wav` (16kHz).
>    - **Sangat Penting:** Karena CQT dan LFCC mungkin memiliki dimensi *frequency bins* atau *time frames* yang berbeda (tergantung *hop length*), buatkan fungsi interpolasi/resampling menggunakan `torch.nn.functional.interpolate` agar dimensi waktu (time axis) keduanya sama persis.
>    - Lakukan konkatenasi (*concatenate*) matriks CQT dan LFCC pada dimensi frekuensi (sehingga membentuk satu tensor gabungan multi-domain).
> 
> 3. **Modifikasi Back-End AASIST (Adapter Layer):**
>    - Impor arsitektur AASIST (`from models.AASIST import Model`).
>    - Modifikasi AASIST: Hapus/nonaktifkan lapisan `Sinc-Conv` bawaannya.
>    - Buatkan sebuah **Adapter Layer (Conv2D layer sederhana)** yang menerima tensor fusi CQT-LFCC kita dan memproyeksikannya (memetakan *channels* dan dimensinya) agar ukurannya persis sesuai dengan input yang diharapkan oleh lapisan *Graph Attention* pertama dari AASIST.
> 
> 4. **Strategi Dataset & Cross-Evaluation (ASVspoof 5 Lokal):**
>    - **Training & Validation:** Simulasikan pemuatan dataset ASVspoof 2019 Logical Access (LA) dari `/kaggle/input/...`.
>    - **Testing (Stress-Test - ASVspoof 5 Lokal):**
>      - Terapkan pemuatan dataset **ASVspoof 5 secara lokal** sesuai panduan **data.md**.
>      - Terapkan **Stratified Quota Sampling** menggunakan pandas dari file `.tsv` lokal:
>        1. Buat fungsi `get_sampled_dataframe(tsv_path, max_bonafide, samples_per_attack)` untuk membaca berkas `.tsv` (`ASVspoof5.dev.track_1.tsv`) menggunakan `pandas` dengan `sep=' '` dan kolom-kolom: `SPEAKER_ID`, `FLAC_FILE_NAME`, `SPEAKER_GENDER`, `CODEC`, `CODEC_Q`, `CODEC_SEED`, `ATTACK_TAG`, `ATTACK_LABEL`, `KEY`, `TMP`.
>        2. Filter kelas `bonafide` sebanyak `max_bonafide` (misal 2500) dan kelas `spoof` dengan *groupby* `ATTACK_LABEL` lalu ambil `samples_per_attack` per jenis serangan agar seimbang.
>        3. Buat kelas `ASVspoof5Dataset(Dataset)` yang menerima DataFrame hasil sampling dan `audio_dir` (path folder `.flac` lokal seperti `flac_D/` untuk dev).
>        4. Di dalam `__getitem__`, ambil file audio `.flac`, lakukan *padding/truncating* agar panjangnya tepat 64600 samples (sesuai standar input AASIST), ubah kolom `KEY` menjadi label biner (0 untuk `bonafide`, 1 untuk `spoof`), dan kembalikan `(waveform_tensor, label_tensor)`.
>        5. Inisialisasi `DataLoader` untuk evaluasi lintas-dataset ini untuk menghitung EER dan min t-DCF.
> 
> 5. **Metrik Evaluasi (Matrix):**
>    - Hitung **Equal Error Rate (EER)**.
>    - Hitung **min t-DCF** (tandem Detection Cost Function).
> 
> 6. **Training Loop:** Bangun siklus pelatihan PyTorch menggunakan `AdamW` optimizer dan `CosineAnnealingLR`. Tampilkan log *Loss*, Akurasi, EER, dan min t-DCF secara reguler di akhir setiap epoch.
> 
> Berikan komentar dan narasi *Markdown* di setiap sel untuk menjelaskan proses interpolasi penyamaan dimensi antara CQT dan LFCC, serta bagaimana *Adapter Layer* menghubungkannya ke AASIST."
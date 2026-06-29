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
> 4. **Strategi Dataset & Cross-Evaluation:**
>    - **Training & Validation:** Simulasikan pemuatan dataset ASVspoof 2019 Logical Access (LA) dari `/kaggle/input/...`.
>    - **Testing (Stress-Test):** Impor dataset **ASVspoof 5** secara *streaming* via Hugging Face (`datasets.load_dataset("jungjee/asvspoof5", streaming=True)`). Terapkan **Stratified Random Sampling** untuk mengambil maksimal 5.000 sampel pengujian, dengan rasio kelas *Bona fide/Spoof* yang seimbang dan mencakup seluruh *Attack ID* secara merata.
> 
> 5. **Metrik Evaluasi (Matrix):**
>    - Hitung **Equal Error Rate (EER)**.
>    - Hitung **min t-DCF** (tandem Detection Cost Function).
> 
> 6. **Training Loop:** Bangun siklus pelatihan PyTorch menggunakan `AdamW` optimizer dan `CosineAnnealingLR`. Tampilkan log *Loss*, Akurasi, EER, dan min t-DCF secara reguler di akhir setiap epoch.
> 
> Berikan komentar dan narasi *Markdown* di setiap sel untuk menjelaskan proses interpolasi penyamaan dimensi antara CQT dan LFCC, serta bagaimana *Adapter Layer* menghubungkannya ke AASIST."
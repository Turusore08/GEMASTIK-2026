# PANDUAN IMPLEMENTASI BASELINE 3: AASIST MURNI (STATE-OF-THE-ART BASELINE)
**Arsitektur:** AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)
**Peran:** Sebagai batas atas pembanding (*State-of-the-Art Baseline*). Ini adalah fondasi utama yang akan kita buktikan masih memiliki celah (*bottleneck*), yang nantinya akan kita sempurnakan dengan fusi CQT-LFCC dan Atensi Temporal pada metode ke-4 dan ke-5.

---

## INSTRUKSI UNTUK AI AGENT (PROMPT JUPYTER NOTEBOOK GENERATOR)

*(Salin teks di bawah ini dan berikan kepada AI Agent seperti GitHub Copilot, ChatGPT, atau Claude untuk membuatkan file `.ipynb`)*

**Prompt AI Agent:**
> "Kamu adalah seorang Senior AI Engineer spesialis Graph Neural Networks untuk Audio Anti-Spoofing. Saya sedang menyusun Baseline 3 menggunakan arsitektur **AASIST Murni** (berdasarkan paper Jung et al., 2022).
> 
> **Konteks & Syarat Eksekusi (Sangat Penting):** > Saya ingin menjalankan skrip ini di **Kaggle secara mandiri (standalone)**. Jangan gunakan referensi *path* lokal seperti `C:\` atau `D:\`. Seluruh *dependencies*, kode model, dan konfigurasi harus tersedia di dalam file `.ipynb` ini. 
> 
> **Tugas:** Buatkan saya skrip Jupyter Notebook `.ipynb` yang memuat tahapan berikut:
> 
> 1. **Persiapan Environment (Cell 1):** >    - Berikan perintah `!git clone https://github.com/clovaai/aasist.git` untuk mengunduh kode resmi AASIST langsung ke dalam *working directory* Kaggle.
>    - Gunakan `sys.path.append('./aasist')` agar *class* dan *module* AASIST bisa diimpor.
> 2. **Konfigurasi & Inisiasi Model:**
>    - AASIST membutuhkan parameter konfigurasi (seperti `filts`, `gat_dims`, dsb). Tuliskan *dictionary* atau konfigurasi default AASIST secara eksplisit di dalam sebuah *cell*, lalu inisiasi modelnya (`from models.AASIST import Model`).
> 3. **Pemrosesan Data (Front-End Bawaan):** >    - AASIST murni menggunakan *Sinc-filter* bawaan untuk mengekstrak fitur dari audio mentah (*raw waveform*).
>    - Buatkan *Data Loader* yang memuat audio `.wav` (16kHz) dan melakukan *padding/truncating* menjadi tepat **64.600 samples** (sesuai standar input AASIST).
> 4. **Strategi Dataset & Cross-Evaluation (ASVspoof 5 Lokal):**
>    - **Training & Validation (ASVspoof 5 Lokal):** Gunakan skema data dari dataset lokal ASVspoof 5 dengan file protokol `ASVspoof5.train.tsv` (untuk training) dan `ASVspoof5.dev.track_1.tsv` (untuk validation/dev) menggunakan teknik **Stratified Quota Sampling** dari `data.md` untuk menghemat memori.
>    - **Testing (Stress-Test - ASVspoof 5 Lokal):**
>      - Terapkan pemuatan dataset **ASVspoof 5 secara lokal** (contoh path: `/kaggle/input/datasets/raditya0/asvspoof5` dengan fallback ke `D:/Lomba/Gemastik 2026/Data Mining/Dataset_ASVspoof5_Sampled`) sesuai panduan **data.md**.
>      - Terapkan **Stratified Quota Sampling** menggunakan pandas dari file `.tsv` lokal:
>        1. Buat fungsi `get_sampled_dataframe(tsv_path, max_bonafide, samples_per_attack)` untuk membaca berkas `.tsv` (`ASVspoof5.dev.track_1.tsv`) menggunakan `pandas` dengan `sep=' '` dan kolom-kolom: `SPEAKER_ID`, `FLAC_FILE_NAME`, `SPEAKER_GENDER`, `CODEC`, `CODEC_Q`, `CODEC_SEED`, `ATTACK_TAG`, `ATTACK_LABEL`, `KEY`, `TMP`.
>        2. Filter kelas `bonafide` sebanyak `max_bonafide` (misal 2500) dan kelas `spoof` dengan *groupby* `ATTACK_LABEL` lalu ambil `samples_per_attack` per jenis serangan agar seimbang.
>        3. Buat kelas `ASVspoof5Dataset(Dataset)` yang menerima DataFrame hasil sampling dan `audio_dir` (path root folder audio). Di dalam `__init__`, gunakan `os.walk(audio_dir)` untuk memindai seluruh subfolder secara rekursif (misalnya `flac_D_aa/flac_D/`, `flac_D_ab/flac_D/`, dll.) dan bangun map/dictionary nama file ke path absolutnya.
>        4. Di dalam `__getitem__`, ambil file audio `.flac` menggunakan path absolut dari map pemetaan tersebut, lakukan *padding/truncating* agar panjangnya tepat 64600 samples (sesuai standar input AASIST), ubah kolom `KEY` menjadi label biner (0 untuk `bonafide`, 1 untuk `spoof`), dan kembalikan `(waveform_tensor, label_tensor)`.
>        5. Inisialisasi `DataLoader` untuk evaluasi lintas-dataset ini untuk menghitung EER dan min t-DCF.
> 5. **Metrik Evaluasi (Matrix):** >    - Hitung dan cetak **Equal Error Rate (EER)** menggunakan `scipy.optimize.brentq` dan `roc_curve`.
>    - Hitung **min t-DCF** (tandem Detection Cost Function). Sertakan fungsi kalkulasinya secara langsung di dalam *notebook* (atau *import* jika tersedia dari repo AASIST).
> 6. **Training Loop:** Buat *loop* pelatihan menggunakan `Adam` optimizer, `CosineAnnealingLR` scheduler, dan *loss function* (Cross Entropy dengan *weight* tertentu jika diperlukan). Log *Loss*, EER, dan min t-DCF di setiap akhir *epoch*.
> 
> Tulis dengan rapi, berikan *markdown* penjelasan di setiap *cell*, dan pastikan struktur kodenya *bug-free* untuk dijalankan langsung di kernel Kaggle."
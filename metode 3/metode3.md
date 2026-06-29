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
> 4. **Strategi Dataset & Cross-Evaluation:**
>    - **Training & Validation:** Simulasikan pemuatan dari subset ASVspoof 2019 Logical Access (LA) menggunakan *path* `/kaggle/input/...`.
>    - **Testing (Stress-Test):** Buat satu *cell* terpisah yang mengimpor dataset **ASVspoof 5** via Hugging Face (`datasets.load_dataset("jungjee/asvspoof5", streaming=True)`). Buatkan fungsi *Stratified Sampling* untuk mengambil maksimal 5.000 sampel pengujian dengan rasio *Bona fide* dan *Spoof* yang merata berdasarkan *metadata/Attack ID*.
> 5. **Metrik Evaluasi (Matrix):** >    - Hitung dan cetak **Equal Error Rate (EER)** menggunakan `scipy.optimize.brentq` dan `roc_curve`.
>    - Hitung **min t-DCF** (tandem Detection Cost Function). Sertakan fungsi kalkulasinya secara langsung di dalam *notebook* (atau *import* jika tersedia dari repo AASIST).
> 6. **Training Loop:** Buat *loop* pelatihan menggunakan `Adam` optimizer, `CosineAnnealingLR` scheduler, dan *loss function* (Cross Entropy dengan *weight* tertentu jika diperlukan). Log *Loss*, EER, dan min t-DCF di setiap akhir *epoch*.
> 
> Tulis dengan rapi, berikan *markdown* penjelasan di setiap *cell*, dan pastikan struktur kodenya *bug-free* untuk dijalankan langsung di kernel Kaggle."
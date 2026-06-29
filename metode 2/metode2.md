# PANDUAN IMPLEMENTASI BASELINE 2: RAWNET2 (END-TO-END)
**Arsitektur:** RawNet2 Murni (End-to-End dari Sinyal Audio Mentah)
**Peran:** Sebagai pembanding model *State-of-the-Art* (SOTA) yang tidak menggunakan ekstraksi fitur (*featureless*), untuk membuktikan bahwa tanpa fusi fitur khusus, model akan kesulitan melakukan generalisasi pada data baru.

---

## INSTRUKSI UNTUK AI AGENT (PROMPT JUPYTER NOTEBOOK GENERATOR)

*(Salin teks di bawah ini dan berikan kepada AI Agent seperti GitHub Copilot, ChatGPT, atau Claude untuk membuatkan file `.ipynb`)*

**Prompt AI Agent:**
> "Kamu adalah seorang Senior AI Engineer yang ahli di bidang riset Audio Anti-Spoofing. Saya sedang membangun Baseline 2 untuk eksperimen saya menggunakan arsitektur **RawNet2** (berdasarkan paper Tak et al., 2021).
> 
> **Konteks Lokal:** Saya sudah memiliki *clone* repositori GitHub resmi RawNet2. Lokasinya berada di PC saya pada direktori: `D:\Lomba\Gemastik 2026\Data Mining\panduan\github`. 
> 
> **Tugas:** Buatkan saya skrip lengkap dalam bentuk **Jupyter Notebook (.ipynb)** yang *reproducible*. Notebook ini harus memuat tahapan dari pemuatan data audio mentah hingga evaluasi akhir.
> 
> **Syarat dan Spesifikasi Kode:**
> 1. **Reproduksibilitas & Integrasi Lokal:** Awali dengan import `sys` dan gunakan `sys.path.append(r"D:\Lomba\Gemastik 2026\Data Mining\panduan\github")` agar Python dapat mengimpor kelas arsitektur `RawNet2` dari *folder* tersebut secara langsung. Jangan buat arsitektur RawNet2 dari awal, cukup *import* dan panggil *class*-nya. Set `random seed` untuk PyTorch dan NumPy.
> 2. **Pemrosesan Data (End-to-End):** Karena RawNet2 menerima *raw waveform*, tidak perlu ada ekstraksi MFCC/LFCC/CQT. Buat fungsi pemuatan data menggunakan `soundfile` atau `librosa` yang melakukan *padding* atau *truncating* pada audio `.wav` (16kHz) menjadi panjang tetap (misalnya 64.000 *samples* atau sekitar 4 detik) agar bisa masuk ke dalam *batch* tensor 1D.
> 3. **Strategi Dataset:**
>    - **Training & Validation:** Gunakan skema data dari subset ASVspoof 2019 Logical Access (LA).
>    - **Testing / Cross-Dataset Evaluation:** Buatkan satu *cell* khusus untuk memuat dataset **ASVspoof 5** menggunakan `datasets.load_dataset("jungjee/asvspoof5", streaming=True)`. Buatkan fungsi *Stratified Sampling* yang mengambil maksimal 5.000 sampel pengujian dengan proporsi *Bona fide* dan *Spoof* yang seimbang, serta memastikan setiap *Attack ID* pada *metadata* terwakili secara merata.
> 4. **Metrik Evaluasi (Matrix):** >    - Kalkulasi dan tampilkan **Equal Error Rate (EER)** menggunakan kurva ROC.
>    - Kalkulasi **minimum tandem Detection Cost Function (min t-DCF)** sesuai protokol ASVspoof.
> 5. **Training Loop:** Bangun *loop* pelatihan menggunakan `Adam` optimizer (atau optimizer yang direkomendasikan di paper RawNet2) dan *loss function* yang sesuai (biasanya *CrossEntropyLoss*). Cetak (*print*) *Loss*, EER, dan min t-DCF pada akhir setiap *epoch*.
> 
> Tolong berikan penjelasan menggunakan format Markdown di dalam Jupyter Notebook untuk setiap blok kodenya, menjelaskan logika pemotongan audio dan alasan digunakannya *stratified sampling* pada data ASVspoof 5."
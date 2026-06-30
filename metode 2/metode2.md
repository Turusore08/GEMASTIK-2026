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
>    - **Testing / Cross-Dataset Evaluation (ASVspoof 5 Lokal):**
>      - Terapkan pemuatan dataset **ASVspoof 5 secara lokal** sesuai panduan **data.md**.
>      - Definisikan base direktori (contoh: `BASE_DIR = r"D:\Lomba\Gemastik 2026\Data Mining\Dataset_ASVspoof5_Sampled"`).
>      - Terapkan **Stratified Quota Sampling** menggunakan pandas dari file `.tsv` lokal:
>        1. Buat fungsi `get_sampled_dataframe(tsv_path, max_bonafide, samples_per_attack)` untuk membaca berkas `.tsv` (`ASVspoof5.dev.track_1.tsv`) menggunakan `pandas` dengan `sep=' '` dan kolom-kolom: `SPEAKER_ID`, `FLAC_FILE_NAME`, `SPEAKER_GENDER`, `CODEC`, `CODEC_Q`, `CODEC_SEED`, `ATTACK_TAG`, `ATTACK_LABEL`, `KEY`, `TMP`.
>        2. Filter kelas `bonafide` sebanyak `max_bonafide` (misal 2500) dan kelas `spoof` dengan *groupby* `ATTACK_LABEL` lalu ambil `samples_per_attack` per jenis serangan agar seimbang.
>        3. Buat kelas `ASVspoof5Dataset(Dataset)` yang menerima DataFrame hasil sampling dan `audio_dir` (path folder `.flac` lokal seperti `flac_D/` untuk dev).
>        4. Di dalam `__getitem__`, ambil file audio `.flac`, lakukan *padding/truncating* agar panjangnya tepat 64000 samples, ubah kolom `KEY` menjadi label biner (0 untuk `bonafide`, 1 untuk `spoof`), dan kembalikan `(waveform_tensor, label_tensor)`.
>        5. Inisialisasi `DataLoader` untuk evaluasi lintas-dataset ini untuk menghitung EER dan min t-DCF.
> 4. **Metrik Evaluasi (Matrix):** >    - Kalkulasi dan tampilkan **Equal Error Rate (EER)** menggunakan kurva ROC.
>    - Kalkulasi **minimum tandem Detection Cost Function (min t-DCF)** sesuai protokol ASVspoof.
> 5. **Training Loop:** Bangun *loop* pelatihan menggunakan `Adam` optimizer (atau optimizer yang direkomendasikan di paper RawNet2) dan *loss function* yang sesuai (biasanya *CrossEntropyLoss*). Cetak (*print*) *Loss*, EER, dan min t-DCF pada akhir setiap *epoch*.
> 
> Tolong berikan penjelasan menggunakan format Markdown di dalam Jupyter Notebook untuk setiap blok kodenya, menjelaskan logika pemotongan audio dan alasan digunakannya *stratified sampling* pada data ASVspoof 5."
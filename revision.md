# Panduan Retraining (Perbaikan Ground Truth Scrambled)

Dokumen ini berisi panduan *step-by-step* untuk memperbaiki dataset secara fundamental (mengembalikan ground truth ke urutan aslinya sesuai `metadata_mapping.md`) dan melakukan training ulang model, sehingga Anda mendapatkan model bersih yang bebas dari *patch/hack* sementara.

---

## Langkah 1: Perbarui File Ground Truth (`.stm`)
Anda tidak perlu melakukan ektraksi *pickle* (keypoints) ulang, karena koordinat videonya sudah tersimpan di file `Pxx_Sxxx_Rxx` yang tepat. Anda hanya perlu **memperbarui teks** di dalam file `.stm`.

Buat file baru bernama `fix_stm.py` di dalam folder `bisindo-cslr`, salin kode di bawah ini, dan jalankan dengan `python fix_stm.py`:

```python
import re
from pathlib import Path

# 1. Baca metadata_mapping untuk mendapatkan teks yang benar
mapping_path = r"C:\Users\IKAJTK\.gemini\antigravity-ide\knowledge\bisindo_skeleton_preprocessing\artifacts\metadata_mapping.md"
with open(mapping_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

true_map = {}
for l in lines:
    m = re.search(r'\d+\.\s+(.*?)\s+->\s+(S\d\d\d)', l)
    if m:
        true_map[m.group(2)] = m.group(1).strip()

# 2. Update semua file .stm
stm_dir = Path("mslr_iccv2025/datasets/mslr2025")
stm_files = [
    "mslr-groundtruth-train.stm",
    "mslr-groundtruth-dev.stm",
    "mslr-groundtruth-test_sd.stm",
    "mslr-groundtruth-test_si_major.stm",
    "mslr-groundtruth-test_si_minor.stm",
]

for stm_file in stm_files:
    file_path = stm_dir / stm_file
    if not file_path.exists():
        continue
        
    with open(file_path, 'r', encoding='utf-8') as f:
        stm_content = f.readlines()
        
    new_content = []
    for line in stm_content:
        # Cari prefix seperti "P01_S015_R05 1 P01 0.0 1.79769e+308 "
        m = re.search(r'(P\d\d_(S\d\d\d)_R\d\d.*?\d+\.?\d* ).*', line)
        if m:
            prefix = m.group(1)
            sid = m.group(2)
            correct_text = true_map.get(sid, "UNKNOWN")
            new_line = prefix + correct_text + "\n"
            new_content.append(new_line)
        else:
            new_content.append(line)
            
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)
        
print("Semua file .stm telah berhasil diperbarui!")
```

---

## Langkah 2: Update `GROUND_TRUTH_TABLE` di API (`app.py`)
Buka file `app.py` dan timpa variabel `GROUND_TRUTH_TABLE` (sekitar baris 47) dengan versi yang sudah urut abjad berikut ini:

```python
GROUND_TRUTH_TABLE: Dict[str, str] = {
    "S001": "AKU CIUM BAU BADAN DIA",
    "S002": "AKU LIHAT ADA ULAR MASUK KELAS",
    "S003": "AKU NILAI JELEK",
    "S004": "AKU PUSING SERING, AKU HARUS PERIKSA MANA",
    "S005": "APA KAMU PERNAH BACA NOVEL B.INGGRIS",
    "S006": "AYAH SAMA IBU MANA",
    "S007": "BADAN AKU GEMUK TAPI BADAN ADIK KURUS",
    "S008": "BUKU AKU SOBEK GEGARA DIA",
    "S009": "DIA ANAK BAIK SAMPAI BANYAK ORANG SUKA",
    "S010": "DIA MENGEJEK AKU",
    "S011": "GAK BOLEH PULANG SEKARANG KAMU",
    "S012": "GIMANA IBUMU BAIK-BAIK ATAU TIDAK",
    "S013": "IBU AKU PUNYA KUCING SAMA IKAN",
    "S014": "KAKAK AKU KASIH HADIAH BUAT AKU",
    "S015": "KAMU BELAJAR BISINDO KAPAN",
    "S016": "KAMU PERGI KEMANA",
    "S017": "KAMU PUNYA ANGGOTA KELUARGA BERAPA",
    "S018": "KENAPA KAMU GAK MASUK KULIAH KEMARIN",
    "S019": "KITA ISTIRAHAT JAM BERAPA",
    "S020": "OBAT BISA BELI TOKO OBAT MANA",
    "S021": "ORANG JAHAT SANA PUKUL AKU BERULANG",
    "S022": "POLISI SANA PUKUL PENCURI",
    "S023": "RUMAH DIMANA KAMU",
    "S024": "SANA BERITA SUDAH BANYAK RIBUAN ORANG LIHAT",
    "S025": "SANA ENAK NASI PADANG TAPI MAHAL",
    "S026": "SANA TOILET KOTOR",
    "S027": "SEPATU DIA KOTOR",
    "S028": "TONG SAMPAH ADA SEMUT BANYAK",
    "S029": "ULANG TAHUN SELAMAT",
    "S030": "ULAR SANA MAKAN KAMBING",
}
```

---

## Langkah 3: Hapus Patch `CORRECTION_MAP` 
Karena model yang baru nanti akan menghasilkan output kalimat yang 100% benar, "jembatan penerjemah" yang saya buat sebelumnya harus dicabut agar tidak menyebabkan salah terjemah terbalik.

Buka `inference/cslr_runner.py`, cari fungsi `_decode_prediction`, dan hapus seluruh blok `CORRECTION_MAP` sehingga hanya tersisa pengembalian string normal seperti sedia kala:

```python
    def _decode_prediction(self, ret_dict: dict, key: str = "recognized_sents_fusion") -> str:
        # ... (kode awal biarkan sama) ...
        
        words = []
        for item in sent:
            if isinstance(item, (list, tuple)):
                words.append(str(item[0]))
            else:
                words.append(str(item))

        # Hapus blok CORRECTION_MAP yang tadi saya tambahkan.
        # Langsung return raw string-nya:
        return " ".join(words)
```

---

## Langkah 4: Mulai Retraining!
Karena file ground truth (`.stm`) sudah benar, Anda tinggal memulai kembali script training Anda (biasanya `main.py` atau `train.py` di dalam folder `mslr_iccv2025` dengan config yaml yang Anda gunakan). 

Model akan mulai menghubungkan fitur video `S015` ke teks `KAMU BELAJAR BISINDO KAPAN` secara alami, tanpa campur aduk alfabetis!

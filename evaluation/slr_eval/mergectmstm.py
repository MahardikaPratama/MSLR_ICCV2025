#!/usr/bin/env python

# Import modul sys untuk mengambil argumen dari command line
import sys

# Ambil nama file CTM dan STM dari argumen command line
ctmFile = sys.argv[1]
stmFile = sys.argv[2]

# Buka file CTM dan STM untuk dibaca
ctm = open(ctmFile, "r")
stm = open(stmFile, "r")

# Inisialisasi list untuk menyimpan baris CTM dan STM
ctmDict = []
stmDict = []
# Variabel untuk menghitung jumlah baris yang ditambahkan ke CTM
addedlines = 0

# Baca seluruh baris dari file CTM, split per kata, dan simpan ke ctmDict
for idx, line in enumerate(ctm):
    l = line.strip().split()
    ctmDict.append(l)

# Baca seluruh baris dari file STM, split per kata, dan simpan ke stmDict
for idx, line in enumerate(stm):
    l = line.strip().split()
    stmDict.append(l)
    # Jika baris CTM dan STM pada index yang sama memiliki ID yang sama
    if len(ctmDict) > idx + addedlines and ctmDict[idx + addedlines][0] == l[0]:  # ctm dan stm cocok
        # Cek apakah ada baris CTM berikutnya yang juga memiliki ID yang sama (multi-line per ID)
        if len(ctmDict) > idx + addedlines + 1:
            while (len(ctmDict) > idx + addedlines + 1) and (ctmDict[idx + addedlines + 1][0] == l[0]):
                addedlines += 1  # Lewati baris CTM tambahan untuk ID yang sama
    else:
        # Jika tidak cocok, tambahkan baris kosong (dummy) ke CTM agar jumlah baris sama dengan STM
        ctmDict.insert(idx + addedlines, [l[0], "1", "0.000", "0.030", "[EMPTY]"])

# Tutup file STM dan CTM setelah selesai membaca
stm.close()
ctm.close()
# Buka kembali file CTM untuk menulis hasil akhir
ctm = open(ctmFile, "w+")

# Tulis ulang seluruh isi ctmDict ke file CTM, setiap baris digabung dengan spasi
for l in ctmDict:
    ctm.write(" ".join(l) + "\n")
ctm.close()

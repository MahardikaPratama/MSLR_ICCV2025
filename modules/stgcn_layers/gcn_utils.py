import torch
import numpy as np
import torch.nn as nn
import math
import copy


class Graph:
    """
    Graph

    Deskripsi:
        Representasi graf skeleton yang menghasilkan adjacency matrix `A`
        berdasarkan `layout` topologi dan `strategy` partisi tetangga.
        Digunakan oleh ST-GCN untuk menentukan konektivitas antar joint
        dan pembobotan lokal pada operasi graph convolution.

        Tiga layout didukung:
            - 'custom_hand21' : 21 keypoint tangan (MediaPipe Hands)
            - 'custom_body'   : 25 keypoint tubuh (MediaPipe Pose)
            - 'custom_mouth_8': 19 keypoint mulut (kontur ring tertutup)

        Tiga strategi partisi adjacency didukung (mengikuti paper ST-GCN):
            - 'uniform'  : semua tetangga satu subset, bobot sama
            - 'distance' : dibagi per jarak hop (K subset = max_hop+1)
            - 'spatial'  : dibagi root / centripetal / centrifugal

    Input (constructor):
        - layout   (str) : nama topologi skeleton yang digunakan.
        - strategy (str) : strategi partisi adjacency matrix.
        - max_hop  (int) : jarak hop maksimum tetangga yang dipertimbangkan.
        - dilation (int) : spasi antar hop (untuk kompatibilitas kernel temporal).

    Proses:
        1. get_edge(layout)       → bangun self.edge dan self.num_node
        2. get_hop_distance(...)  → hitung matriks jarak hop self.hop_dis
        3. get_adjacency(strategy)→ bentuk dan normalisasi self.A

    Output (atribut publik):
        - self.A        : numpy array (K, V, V) adjacency matrix ternormalisasi.
        - self.edge     : list pasangan (i, j) yang merepresentasikan edges.
        - self.num_node : jumlah node/joint (V) pada layout.
    """

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1):
        # simpan max_hop sebagai atribut; dipakai di get_adjacency via valid_hop
        self.max_hop = max_hop
        # simpan dilation sebagai atribut; dipakai di valid_hop range step
        self.dilation = dilation

        # bangun daftar edge dan jumlah node sesuai layout yang dipilih
        self.get_edge(layout)
        # hitung jarak hop minimum antar semua pasangan node
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        # bentuk adjacency matrix (K,V,V) sesuai strategi partisi
        self.get_adjacency(strategy)

    def __str__(self):
        # kembalikan adjacency matrix saat objek di-print untuk debugging
        return self.A


    def get_edge(self, layout):
        """
        Deskripsi:
            Membangun topologi graf skeleton sesuai nama layout yang diberikan.
            Mendefinisikan node (joint), edge (koneksi antar joint), dan
            node pusat (center) yang dipakai oleh strategi partisi 'spatial'.

        Input:
            - layout (str): nama layout, salah satu dari
              ['custom_hand21', 'custom_body', 'custom_mouth_8'].

        Proses:
            - Tentukan self.num_node sesuai jumlah keypoint layout.
            - Buat self_link: list self-loop (i, i) untuk semua node.
            - Buat neighbor_1base: list koneksi anatomis antar joint.
            - Gabungkan keduanya menjadi self.edge.
            - Tetapkan self.center sebagai indeks node pusat skeleton.

        Output:
            - self.num_node (int)       : jumlah node/joint.
            - self.edge (list of tuple) : semua edge termasuk self-loop.
            - self.center (int)         : indeks node pusat skeleton.
        """

        if layout == 'custom_hand21':
            # tangan memiliki 21 keypoint: 1 wrist + 4 sendi x 5 jari
            self.num_node = 21
            # buat self-loop untuk setiap node agar fitur joint itu sendiri
            # ikut diagregasi saat convolution (setara dengan +I pada A+I di paper)
            self_link = [(i, i) for i in range(self.num_node)]
            # definisikan koneksi anatomis antar joint tangan mengikuti
            # struktur MediaPipe Hands: wrist(0) → tiap jari → ujung jari
            neighbor_1base = [
                # ibu jari: wrist → MCP → PIP → DIP → tip
                [0, 1], [1, 2], [2, 3], [3, 4],
                # telunjuk: wrist → MCP → PIP → DIP → tip
                [0, 5], [5, 6], [6, 7], [7, 8],
                # jari tengah: wrist → MCP → PIP → DIP → tip
                [0, 9], [9, 10], [10, 11], [11, 12],
                # jari manis: wrist → MCP → PIP → DIP → tip
                [0, 13], [13, 14], [14, 15], [15, 16],
                # kelingking: wrist → MCP → PIP → DIP → tip
                [0, 17], [17, 18], [18, 19], [19, 20],
            ]
            # tidak ada preprocessing tambahan, langsung pakai sebagai neighbor
            neighbor_link = neighbor_1base
            # gabungkan self-loop dan koneksi antar joint menjadi edge list lengkap
            self.edge = self_link + neighbor_link
            # wrist (index 0) sebagai pusat/root skeleton tangan
            self.center = 0

        elif layout == 'custom_body':
            # tubuh memiliki 25 keypoint sesuai output MediaPipe Pose
            self.num_node = 25
            # buat self-loop untuk semua 25 node
            self_link = [(i, i) for i in range(self.num_node)]
            # definisikan koneksi anatomis tubuh: wajah, lengan, dan badan
            neighbor_1base = [
                # wajah bagian kiri: nose → left_eye_inner → left_eye → left_eye_outer → left_ear
                [0, 1], [1, 2], [2, 3], [3, 7],
                # wajah bagian kanan: nose → right_eye_inner → right_eye → right_eye_outer → right_ear
                [0, 4], [4, 5], [5, 6], [6, 8],
                # mulut: mouth_left ↔ mouth_right
                [9, 10],
                # lengan kiri: shoulder → elbow → wrist → pinky → index → thumb
                [11, 13], [13, 15], [15, 17], [17, 19],
                # koneksi silang pergelangan kiri (pinky ↔ thumb dan wrist ↔ thumb)
                [15, 19], [15, 21],
                # lengan kanan: shoulder → elbow → wrist → pinky → index → thumb
                [12, 14], [14, 16], [16, 18], [18, 20],
                # koneksi silang pergelangan kanan
                [16, 20], [16, 22],
                # badan: left_shoulder → left_hip, right_shoulder → right_hip,
                #        left_hip ↔ right_hip
                [11, 23], [12, 24], [23, 24],
            ]
            # langsung pakai sebagai neighbor tanpa preprocessing
            neighbor_link = neighbor_1base
            # gabungkan self-loop dan koneksi anatomis
            self.edge = self_link + neighbor_link
            # nose (index 0) sebagai pusat/root skeleton tubuh
            self.center = 0

        elif layout == 'custom_mouth_8':
            # mulut memiliki 19 keypoint kontur bibir
            self.num_node = 19
            # buat self-loop untuk semua 19 node
            self_link = [(i, i) for i in range(self.num_node)]
            # bangun koneksi ring tertutup: 0→1→2→...→18→0
            # karena bibir adalah kontur tertutup, bukan rantai terbuka
            neighbor_1base = (
                # chain linear dari node 0 sampai 17
                [[i, i + 1] for i in range(self.num_node - 1)]
                # tutup ring: node terakhir (18) kembali ke node pertama (0)
                + [[self.num_node - 1, 0]]
            )
            # langsung pakai sebagai neighbor
            neighbor_link = neighbor_1base
            # gabungkan self-loop dan koneksi ring
            self.edge = self_link + neighbor_link
            # titik referensi tengah di kontur bibir (bukan index 0)
            self.center = 2


    def get_adjacency(self, strategy):
        """
        Deskripsi:
            Membangun dan menyimpan adjacency matrix `self.A` dalam format
            (K, V, V) sesuai strategi partisi yang dipilih. Mengikuti tiga
            strategi partisi dari paper ST-GCN (Yan et al., 2018):
                - 'uniform'  : K=1, semua tetangga satu subset
                - 'distance' : K=max_hop+1, dipisah per jarak hop
                - 'spatial'  : K=3 (untuk max_hop=1), dipisah berdasarkan
                               posisi relatif terhadap pusat skeleton

        Input:
            - strategy (str): strategi partisi, salah satu dari
              ['uniform', 'distance', 'spatial'].

        Proses:
            1. Tentukan valid_hop dari range(0, max_hop+1, dilation).
            2. Bangun adjacency biner dari hop_dis lalu normalisasi.
            3. Bergantung strategy:
               - uniform  : satu matrix, semua hop digabung.
               - distance : K matrix, tiap matrix untuk satu nilai hop.
               - spatial  : tiap hop dipecah ke a_root/a_close/a_further
                            berdasarkan jarak node ke self.center.
            4. Stack semua matrix → self.A shape (K, V, V).

        Output:
            - self.A (numpy array, shape K×V×V): adjacency matrix
              ternormalisasi siap dipakai oleh ST-GCN layer.
        """

        # tentukan hop yang valid: [0, 1] untuk max_hop=1, dilation=1
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        # inisialisasi adjacency biner V×V dengan semua nol
        adjacency = np.zeros((self.num_node, self.num_node))
        # isi posisi [i,j] dengan 1 jika jarak hop-nya masuk valid_hop
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        # normalisasi adjacency agar kolom berjumlah 1 (degree normalization)
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            # strategi paling sederhana: semua tetangga diperlakukan sama
            # K=1 sehingga hanya ada satu matrix adjacency
            A = np.zeros((1, self.num_node, self.num_node))
            # isi satu-satunya slice dengan seluruh adjacency ternormalisasi
            A[0] = normalize_adjacency
            # simpan ke atribut, siap diambil oleh CoSign2s
            self.A = A

        elif strategy == 'distance':
            # strategi jarak: pisahkan tetangga per nilai hop
            # K = jumlah hop valid (biasanya 2: hop-0 dan hop-1)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                # isi slice ke-i hanya dengan nilai dari jarak hop tertentu
                # posisi lain tetap nol → setiap slice = satu subset tetangga
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            # simpan ke atribut
            self.A = A

        elif strategy == 'spatial':
            # strategi spasial: bagi tetangga berdasarkan posisi relatif ke center
            # mengikuti persamaan (8) paper: root / centripetal / centrifugal
            A = []
            for hop in valid_hop:
                # inisialisasi tiga subset kosong untuk hop ini
                a_root    = np.zeros((self.num_node, self.num_node))
                a_close   = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        # hanya proses pasangan (j,i) yang berjarak tepat `hop`
                        if self.hop_dis[j, i] == hop:

                            if (self.hop_dis[j, self.center]
                                    == self.hop_dis[i, self.center]):
                                # j dan i sama jauhnya dari center → subset root
                                a_root[j, i] = normalize_adjacency[j, i]

                            elif (self.hop_dis[j, self.center]
                                    > self.hop_dis[i, self.center]):
                                # j lebih jauh dari center daripada i
                                # → j bergerak menjauh (centripetal dari sudut i)
                                a_close[j, i] = normalize_adjacency[j, i]

                            else:
                                # j lebih dekat ke center daripada i
                                # → j bergerak ke arah center (centrifugal dari i)
                                a_further[j, i] = normalize_adjacency[j, i]

                if hop == 0:
                    # self-loop: hanya ada subset root (tidak ada tetangga)
                    A.append(a_root)
                else:
                    # hop > 0: dua subset — gabungan root+close, dan further
                    A.append(a_root + a_close)
                    A.append(a_further)

            # stack list matrix menjadi array 3D (K, V, V)
            A = np.stack(A)
            # simpan ke atribut
            self.A = A

        else:
            # strategi tidak dikenal → lempar error eksplisit
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    """
    Deskripsi:
        Menghitung jarak hop minimum antar semua pasangan node pada graf
        tak berarah. Digunakan oleh Graph.get_adjacency() untuk menentukan
        subset tetangga sesuai strategi partisi.

    Input:
        - num_node (int)         : jumlah node (V) pada graf.
        - edge (list of tuple)   : daftar pasangan (i, j) yang merepresentasikan
                                   edges (termasuk self-loop).
        - max_hop (int)          : jarak hop maksimum yang dihitung; node di
                                   luar radius ini diberi nilai inf.

    Proses:
        1. Bangun adjacency biner A dari daftar edge (dua arah).
        2. Hitung matrix power A^d untuk d = 0..max_hop.
        3. Konversi ke boolean arrive_mat: True jika ada path d-hop.
        4. Loop dari max_hop turun ke 0, assign nilai d ke hop_dis
           sehingga nilai terkecil (jarak terpendek) yang tersimpan.

    Output:
        - hop_dis (numpy array, shape V×V): matriks jarak hop minimum.
          Nilai 0 = self-loop, 1 = tetangga langsung, inf = tidak terhubung
          dalam radius max_hop.
    """

    # inisialisasi adjacency biner V×V dengan semua nol
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        # isi dua arah karena graf tak berarah (undirected)
        A[j, i] = 1
        A[i, j] = 1

    # inisialisasi matriks jarak dengan infinity (belum ada path yang diketahui)
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    # hitung matrix power A^0, A^1, ..., A^max_hop
    # A^d[i,j] > 0 berarti ada path dari i ke j dalam tepat d langkah
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    # konversi ke boolean: True jika bisa dicapai dalam d hop
    arrive_mat = np.stack(transfer_mat) > 0
    # loop terbalik dari max_hop ke 0 agar nilai lebih kecil menimpa yang besar
    # sehingga yang tersimpan adalah jarak terpendek (bukan terpanjang)
    for d in range(max_hop, -1, -1):
        # assign nilai d ke semua posisi yang bisa dicapai dalam d hop
        hop_dis[arrive_mat[d]] = d
    # kembalikan matriks jarak hop minimum V×V
    return hop_dis


def normalize_digraph(A):
    """
    Deskripsi:
        Menormalisasi adjacency matrix dengan pembagian per-kolom (degree
        normalization) sehingga tiap kolom berjumlah 1 apabila memungkinkan.
        Mengikuti konvensi normalisasi lokal yang digunakan paper ST-GCN
        (Yan et al., 2018), sebagai penyederhanaan dari Λ^{-1/2}(A+I)Λ^{-1/2}.

    Input:
        - A (numpy array, shape V×V): adjacency matrix biner atau berbobot.

    Proses:
        1. Hitung degree tiap kolom: Dl = sum(A, axis=0), shape (V,).
        2. Bangun matriks diagonal Dn dengan Dl[i]^{-1} pada diagonal;
           lewati jika Dl[i] = 0 untuk mencegah pembagian dengan nol.
        3. Kalikan A dengan Dn: AD = A @ Dn.
           Efeknya: tiap kolom j dibagi dengan degree[j].

    Output:
        - AD (numpy array, shape V×V): adjacency matrix ternormalisasi
          di mana kontribusi tiap node diskalakan oleh jumlah koneksinya.
    """

    # hitung degree tiap kolom: berapa banyak edge masuk ke tiap node
    Dl = np.sum(A, 0)
    # ambil jumlah node dari shape matriks
    num_node = A.shape[0]
    # inisialisasi matriks diagonal D^{-1} dengan semua nol
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            # isi diagonal dengan invers degree; lewati node terisolasi (degree=0)
            # untuk mencegah division by zero
            Dn[i, i] = Dl[i] ** (-1)
    # kalikan A dengan D^{-1}: normalisasi kolom
    # hasilnya: tiap kolom j dari A dibagi dengan degree[j]
    AD = np.dot(A, Dn)
    # kembalikan adjacency ternormalisasi
    return AD
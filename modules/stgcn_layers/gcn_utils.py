import torch
import numpy as np
import torch.nn as nn
import math
import copy


class Graph:
    """
    Representation of the skeleton graph yielding adjacency matrix `A`
    based on topology `layout` and neighbor partition `strategy`.

    Parameters
    ----------
    layout : str
        Name of the skeleton topology ('custom_hand21', 'custom_body', 'custom_mouth_8').
    strategy : str
        Adjacency partition strategy ('uniform', 'distance', 'spatial').
    max_hop : int
        Maximum hop distance for neighbors.
    dilation : int
        Spacing between hops.
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
        Build skeleton topology based on the given layout.

        Parameters
        ----------
        layout : str
            Layout name ('custom_hand21', 'custom_left_hand', 'custom_right_hand', 'custom_body', 'custom_mouth_8').
        """

        if layout in ('custom_hand21', 'custom_left_hand', 'custom_right_hand'):
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

        elif layout == 'custom_left_arm':
            self.num_node = 3
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1], [1, 2]
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'custom_right_arm':
            self.num_node = 3
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1], [1, 2]
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
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
        Build and store adjacency matrix `self.A` according to the partition strategy.

        Parameters
        ----------
        strategy : str
            Partition strategy ('uniform', 'distance', 'spatial').
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
    Compute the minimum hop distance between all pairs of nodes.

    Parameters
    ----------
    num_node : int
        Number of nodes in the graph.
    edge : list
        List of edge tuples (i, j).
    max_hop : int
        Maximum calculated hop distance.

    Returns
    -------
    ndarray
        Hop distance matrix with shape (num_node, num_node).
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
    Normalize the adjacency matrix using degree normalization (column-wise).

    Parameters
    ----------
    A : ndarray
        Binary or weighted adjacency matrix with shape (V, V).

    Returns
    -------
    ndarray
        Normalized adjacency matrix.
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
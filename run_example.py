# run_example.py
import json
from torch.utils.data import DataLoader
from datasets.skeleton_feeder import SkeletonFeeder

# load gloss->index map (sesuaikan path)
with open("datasets/mslr2025/global_gloss_dict.json","r",encoding="utf-8") as f:
    g = json.load(f)["gloss2id"]
g2i = {k: v["index"] for k, v in g.items()}

ds = SkeletonFeeder(
    gloss_dict=g2i,
    mode="dev",
    transform_mode=False,
    datatype="skeleton",
    dataset_root="datasets/mslr2025",
    used_part=["hand21"],
    split=[21,42],
    norm_point=[0,21],
    normalization_types=["spatial"],
    downsampling=False,
)

# single sample via __getitem__
sample = ds[0]
print("sample types:", type(sample[0]), type(sample[1]), type(sample[2]))
print("sample[0] shape (T, K, C):", sample[0].shape)  # T frames, K keypoints, C channels (expect C=7)

# batch via DataLoader (collate_fn dari feeder)
loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
batch = next(iter(loader))
print("batch keys:", batch.keys())
print("x shape (B, T_padded, K, C):", batch['x'].shape)
print("len_x shape:", batch['len_x'].shape)
print("label shape (flattened):", batch['label'].shape)
print("label_lgt (per-sample lengths):", batch['label_lgt'])
print("origin_info (metadata):", batch['origin_info'][:2])
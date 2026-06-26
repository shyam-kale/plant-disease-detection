"""
leaf_dataset.py
---------------
Module-level PyTorch Dataset for spinach leaf images.
Must be a SEPARATE top-level module (not defined inside a function or try/except)
so Windows multiprocessing can pickle it correctly.
"""
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image as _PIL

    class LeafDataset(Dataset):
        """
        Spinach leaf image dataset.
        Defined at module level in a dedicated file so Windows pickle works.
        """
        def __init__(self, paths: list, labels: list, tfm, label_to_idx: dict):
            self.paths        = list(paths)
            self.labels       = list(labels)
            self.tfm          = tfm
            self.label_to_idx = label_to_idx

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                img = _PIL.open(self.paths[i]).convert("RGB")
            except Exception:
                img = _PIL.new("RGB", (380, 380), 0)
            return self.tfm(img), self.label_to_idx.get(self.labels[i], 0)

except ImportError:
    # PyTorch not installed — placeholder so imports don't break
    class LeafDataset:  # type: ignore
        def __init__(self, *a, **kw): raise RuntimeError("PyTorch not installed")

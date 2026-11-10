import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

dataset_dir = Path(r"c:\CropGuard v3\augmented_plantdoc")
aug_pattern = re.compile(r'_aug_.*$', re.IGNORECASE)

records = []
seen_files = set()

for split in ["train", "val", "test"]:
    sdir = dataset_dir / split
    if not sdir.exists(): continue
    for cdir in sdir.iterdir():
        if not cdir.is_dir(): continue
        cname = cdir.name
        for f in cdir.iterdir():
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                target_cname = cname
                if f.name == "Potato leaf late blight (73)_aug_p0_1.jpg" and cname == "Potato leaf early blight":
                    target_cname = "Potato leaf late blight"
                
                key = (target_cname, f.name)
                if key in seen_files:
                    continue
                seen_files.add(key)
                
                base_id = aug_pattern.sub('', f.stem).strip()
                records.append({
                    'class': target_cname,
                    'base_id': base_id,
                    'filename': f.name,
                    'path': f
                })

classes = sorted(list(set(r['class'] for r in records)))
seed = 47

for c in classes:
    c_records = [r for r in records if r['class'] == c]
    groups = [r['base_id'] for r in c_records]
    
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, rem_idx = next(gss_train.split(c_records, groups=groups))
    
    rem_records = [c_records[i] for i in rem_idx]
    rem_groups = [groups[i] for i in rem_idx]
    
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_sub_idx, test_sub_idx = next(gss_val.split(rem_records, groups=rem_groups))
    
    c_train = [c_records[i] for i in train_idx]
    c_val = [rem_records[i] for i in val_sub_idx]
    c_test = [rem_records[i] for i in test_sub_idx]
    
    train_bases = set(r['base_id'] for r in c_train)
    val_bases = set(r['base_id'] for r in c_val)
    test_bases = set(r['base_id'] for r in c_test)
    
    print(f"{c:30s} | Bases: {len(train_bases):2d}/{len(val_bases):2d}/{len(test_bases):2d} | Imgs: {len(c_train):3d}/{len(c_val):3d}/{len(c_test):3d} (tot={len(c_records)})")

import os
import re
from pathlib import Path
from collections import defaultdict, Counter

dataset_dir = Path(r"c:\CropGuard v3\augmented_plantdoc")
aug_pattern = re.compile(r'_aug_.*$', re.IGNORECASE)

records = []
for split in ["train", "val", "test"]:
    split_dir = dataset_dir / split
    if not split_dir.exists():
        continue
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        cname = class_dir.name
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                stem = f.stem
                base_stem = aug_pattern.sub('', stem).strip()
                records.append({
                    'class': cname,
                    'split': split,
                    'filename': f.name,
                    'stem': stem,
                    'base_stem': base_stem,
                    'path': f
                })

print(f"Total image records scanned: {len(records)}")

# Check duplicate files by filename
by_filename = defaultdict(list)
for r in records:
    by_filename[r['filename']].append(r)

dups_same_class = 0
dups_diff_class = 0
for fname, items in by_filename.items():
    classes = set(item['class'] for item in items)
    if len(classes) > 1:
        dups_diff_class += 1
        print(f"Duplicate filename in different classes: {fname} -> {classes}")
    elif len(items) > 1:
        dups_same_class += 1

print(f"Filenames appearing multiple times in same class: {dups_same_class}")
print(f"Filenames appearing in multiple classes: {dups_diff_class}")

# What if we deduplicate by (class, filename)?
unique_records = {}
for r in records:
    key = (r['class'], r['filename'])
    if key not in unique_records:
        unique_records[key] = r

print(f"Total unique (class, filename) pairs: {len(unique_records)}")

# Check base_stems across classes
base_to_classes = defaultdict(set)
for (cname, fname), r in unique_records.items():
    base_to_classes[r['base_stem']].add(cname)

cross_class_bases = {b: cs for b, cs in base_to_classes.items() if len(cs) > 1}
print(f"Base stems appearing in multiple classes: {len(cross_class_bases)}")
for b, cs in cross_class_bases.items():
    print(f"  Base '{b}': {cs}")
    # Print the filenames
    for c in cs:
        matching = [r['filename'] for (cn, fn), r in unique_records.items() if cn == c and r['base_stem'] == b]
        print(f"    In {c}: {matching}")

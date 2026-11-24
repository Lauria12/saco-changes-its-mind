"""Dataset loading. Imagenette/CIFAR can self-download; ImageNet cannot (no stable
public direct-download URL) and must be staged by the user first -- see
stage_imagenet_val() below, called from prepare_data.py, which reads directly out
of a Kaggle "ImageNet Object Localization Challenge" zip archive available on
Snellius's shared datasets (no download needed).

Snellius compute nodes are not guaranteed to have internet access -- run
prepare_data.py on the `staging` partition (or the login node, for anything quick)
before submitting GPU jobs, so downloads below become no-ops on the compute node.
"""
import csv
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from torchvision import datasets

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

IMAGENET_KAGGLE_ZIP = (
    "/projects/2/managed_datasets/imagenet_raw/raw/"
    "imagenet-object-localization-challenge.zip"
)


def _download_and_extract_imagenette(data_root: Path) -> Path:
    archive_path = data_root / "imagenette2-160.tgz"
    extract_dir = data_root / "imagenette2-160"
    if extract_dir.exists():
        return extract_dir
    data_root.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        urllib.request.urlretrieve(IMAGENETTE_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_root)
    return extract_dir


def stage_imagenet_val(data_root: Path, zip_path: str = IMAGENET_KAGGLE_ZIP) -> Path:
    """Stages the ImageNet-1k VALIDATION split only (this pipeline never requests
    split="train" for any dataset -- confirmed by grep across run_experiment.py/
    prepare_data.py -- so the ~150GB train set isn't needed).

    Reads directly out of the Kaggle "ImageNet Object Localization Challenge" zip
    on Snellius's shared datasets (no download, no HF account/gated-dataset auth
    needed) -- raw JPEG bytes are copied as-is (no decode/re-encode), so this is
    fast and introduces no compression-quality loss. Labels come from
    LOC_val_solution.csv (ImageId -> "<synset_id> <bbox...> [<synset_id> <bbox...>]";
    multiple entries are multiple boxes of the SAME class for that image, so only
    the first token is needed). Uses real WordNet synset IDs as folder names, not
    an arbitrary numeric remapping -- torchvision.datasets.ImageFolder sorts class
    folder names alphabetically to assign indices, which is the exact same
    convention torchvision/timm's own ImageNet-pretrained models were trained
    with, so this requires no separate label-index verification step (unlike a
    HF-integer-label-based approach would)."""
    val_dir = data_root / "val"
    if val_dir.exists() and any(val_dir.iterdir()):
        return val_dir

    zf = zipfile.ZipFile(zip_path)
    with zf.open("LOC_val_solution.csv") as f:
        reader = csv.DictReader(line.decode() for line in f)
        label_map = {row["ImageId"]: row["PredictionString"].split()[0] for row in reader}

    val_dir.mkdir(parents=True, exist_ok=True)
    for image_id, synset in label_map.items():
        class_dir = val_dir / synset
        class_dir.mkdir(exist_ok=True)
        with zf.open(f"ILSVRC/Data/CLS-LOC/val/{image_id}.JPEG") as src:
            (class_dir / f"{image_id}.JPEG").write_bytes(src.read())

    n_classes = len(list(val_dir.iterdir()))
    n_images = sum(1 for _ in val_dir.rglob("*.JPEG"))
    print(f"Staged ImageNet val: {n_images} images across {n_classes} classes at {val_dir}")
    return val_dir


def get_dataset(name: str, data_root: str, transform, split: str = "val"):
    root = Path(data_root)

    if name == "imagenette":
        extract_dir = _download_and_extract_imagenette(root)
        return datasets.ImageFolder(extract_dir / split, transform=transform)

    if name == "cifar10":
        return datasets.CIFAR10(root=str(root), train=(split == "train"),
                                 download=True, transform=transform)

    if name == "cifar100":
        return datasets.CIFAR100(root=str(root), train=(split == "train"),
                                  download=True, transform=transform)

    if name == "imagenet":
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected an ImageFolder-structured ImageNet at {split_dir} "
                f"(class-name subdirectories under {split_dir}). ImageNet has no "
                f"auto-download here -- point --data-root at an existing copy."
            )
        return datasets.ImageFolder(split_dir, transform=transform)

    raise ValueError(f"Unknown dataset {name!r}")

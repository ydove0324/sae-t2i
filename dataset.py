import yaml
import os
from torch.utils.data import Dataset
from PIL import Image
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import torch
from torchvision import transforms
from PIL import Image
def load_index_synset_map(index_synset_path: str):
    """
    读取 index_synset.yaml，构造:
        index2synset: Dict[int, str]
        synset2index: Dict[str, int]

    兼容两种常见格式：
    1) 映射: {0: 'n01440764', 1: 'n01443537', ...}
    2) 列表: ['n01440764', 'n01443537', ...]   # 下标就是 index
    """
    with open(index_synset_path, "r") as f:
        data = yaml.safe_load(f)

    index2synset = {}

    if isinstance(data, dict):
        for k, v in data.items():
            idx = int(k)
            index2synset[idx] = str(v)
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            index2synset[idx] = str(v)
    else:
        raise ValueError(f"Unsupported index_synset format: {type(data)}")

    synset2index = {syn: idx for idx, syn in index2synset.items()}
    return index2synset, synset2index

class ImageNetIdxDataset(Dataset):
    """
    从 root 下所有 n* 文件夹中读图像，
    label 通过 index_synset.yaml 映射成 0..N-1 的整数。
    """

    def __init__(self, root: str, index_synset_path: str, transform=None):
        self.root = root
        self.transform = transform

        # 读 index_synset.yaml
        self.index2synset, self.synset2index = load_index_synset_map(index_synset_path)

        self.samples = []  # list of (img_path, label)

        # 遍历 root 下所有子目录
        for entry in os.scandir(root):
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith("n"):
                # 跳过 filelist.txt 等非 synset 目录
                continue

            synset = name
            if synset not in self.synset2index:
                # index_synset.yaml 里没有这个 synset，就跳过
                continue
            label = self.synset2index[synset]

            class_dir = entry.path
            # 收集所有图片文件
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".webp")):
                    continue
                img_path = os.path.join(class_dir, fname)
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image samples found under {root} with index_synset_path={index_synset_path}"
            )

        print(f"[ImageNetIdxDataset] Found {len(self.samples)} images in {len(self.synset2index)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class OverfitSingleImageDataset(Dataset):
    """
    过拟合用：dataset 里只有一张图片（同一张重复 N 次返回）。

    参数：
    - image_path: 单张图片路径
    - length: dataset 的“虚拟长度”（例如 1024，方便你跑很多 step）
    - label: 固定返回的 label（默认 0）
    - transform: torchvision transform
    - cache_in_memory: True 时只读一次图片，后续直接复制（更快更稳定）
    """

    def __init__(
        self,
        image_path: str,
        length: int = 1024,
        label: int = 0,
        transform=None,
        cache_in_memory: bool = True,
    ):
        self.image_path = image_path
        self.length = int(length)
        self.label = int(label)
        self.transform = transform
        self.cache_in_memory = bool(cache_in_memory)

        if self.length <= 0:
            raise ValueError("length must be > 0")
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"image_path not found: {self.image_path}")

        self._cached_pil = None
        if self.cache_in_memory:
            # 只读一次，避免 I/O 抖动
            self._cached_pil = Image.open(self.image_path).convert("RGB")

    def __len__(self):
        return self.length

    def _load_pil(self):
        if self._cached_pil is not None:
            # 返回一个拷贝，避免 transform 或后续操作意外改到缓存对象
            return self._cached_pil.copy()
        return Image.open(self.image_path).convert("RGB")

    def __getitem__(self, idx):
        img = self._load_pil()
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label

class SingleClassDataset(Dataset):
    """
    只加载指定的一个类别（target_class_index）。
    用于微调或测试特定类别。
    """
    def __init__(self, root: str, index_synset_path: str, target_class_index: int, transform=None):
        self.root = root
        self.transform = transform
        self.target_class_index = int(target_class_index)

        # 1. 读取 index -> synset 的映射
        self.index2synset, self.synset2index = load_index_synset_map(index_synset_path)

        # 2. 检查这个 index 是否在 yaml 里
        if self.target_class_index not in self.index2synset:
            raise ValueError(f"Target index {self.target_class_index} not found in {index_synset_path}")

        # 3. 构造该类别的具体路径
        target_synset = self.index2synset[self.target_class_index]
        class_dir = os.path.join(self.root, target_synset)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir} (synset: {target_synset})")

        self.samples = []

        # 4. 只读取该文件夹下的图片
        # 使用 sorted 保证顺序一致
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".webp")):
                img_path = os.path.join(class_dir, fname)
                self.samples.append((img_path, self.target_class_index))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image samples found in {class_dir} for class index {self.target_class_index}"
            )

        print(f"[SingleClassDataset] Found {len(self.samples)} images for class {self.target_class_index} ({target_synset}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

IMG_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    import numpy as np
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),                      # [0,1]
        transforms.Lambda(lambda t: t * 2.0 - 1.0), # [-1,1]
    ])


def load_index_synset_map(index_synset_path: str):
    with open(index_synset_path, "r") as f:
        data = yaml.safe_load(f)

    index2synset = {}
    if isinstance(data, dict):
        for k, v in data.items():
            index2synset[int(k)] = str(v)
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            index2synset[idx] = str(v)
    else:
        raise ValueError(f"Unsupported index_synset format: {type(data)}")

    synset2index = {syn: idx for idx, syn in index2synset.items()}
    return index2synset, synset2index


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def make_cache_tag(vae_ckpt: str, image_size: int, normalize_tag: str = "normalize_sae_v1") -> str:
    # 让不同 VAE/分辨率/归一化方式的缓存互不冲突
    base = f"{os.path.basename(vae_ckpt)}|{image_size}|{normalize_tag}"
    return _sha1(base)[:16]


def sample_cache_relpath(img_path: str, tag: str) -> str:
    # 加入 mtime/size：图片被替换时自动失效
    try:
        st = os.stat(img_path)
        mtime = int(st.st_mtime)
        size = int(st.st_size)
    except FileNotFoundError:
        mtime = 0
        size = 0
    key = _sha1(f"{tag}|{img_path}|mtime={mtime}|size={size}")
    return os.path.join(key[:2], f"{key}.pt")


@dataclass
class DatasetSpec:
    root: Optional[str]
    index_synset_path: Optional[str]
    image_size: int
    vae_ckpt: str
    cache_dir: str
    single_class_index: Optional[int] = None
    overfit_image: Optional[str] = None
    overfit_length: int = 1024


class ImageNetLatentCacheDataset(Dataset):
    """
    行为：
      - __getitem__ 先查 cache：
          命中 -> 返回 {"has_latent": True, "latent": [C,H,W], "label": int, "path": str}
          未命中 -> 返回 {"has_latent": False, "img": [3,H,W], "label": int, "path": str}
    注意：
      - 这里只做 “查缓存 + 读图”，不做 GPU encode
      - 缓存文件格式：torch.save({"latent": Tensor[C,H,W], "label": int, "path": str})
    """
    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.transform = build_transform(spec.image_size)

        self.tag = make_cache_tag(spec.vae_ckpt, spec.image_size)
        self.cache_root = os.path.join(spec.cache_dir, f"latents_{self.tag}")
        os.makedirs(self.cache_root, exist_ok=True)

        # 生成 samples: List[(img_path, label)]
        self.samples: List[Tuple[str, int]] = []

        if spec.overfit_image is not None:
            if not os.path.isfile(spec.overfit_image):
                raise FileNotFoundError(f"overfit_image not found: {spec.overfit_image}")
            for _ in range(int(spec.overfit_length)):
                self.samples.append((spec.overfit_image, 0))

        else:
            if spec.root is None or spec.index_synset_path is None:
                raise ValueError("root and index_synset_path are required for imagenet/single_class modes")

            index2synset, synset2index = load_index_synset_map(spec.index_synset_path)

            if spec.single_class_index is not None:
                idx = int(spec.single_class_index)
                if idx not in index2synset:
                    raise ValueError(f"single_class_index {idx} not in yaml")
                synset = index2synset[idx]
                class_dir = os.path.join(spec.root, synset)
                if not os.path.isdir(class_dir):
                    raise FileNotFoundError(f"class dir not found: {class_dir}")
                for fname in sorted(os.listdir(class_dir)):
                    if fname.lower().endswith(IMG_EXTS):
                        self.samples.append((os.path.join(class_dir, fname), idx))
            else:
                for entry in os.scandir(spec.root):
                    if not entry.is_dir():
                        continue
                    synset = entry.name
                    if not synset.startswith("n"):
                        continue
                    if synset not in synset2index:
                        continue
                    label = synset2index[synset]
                    for fname in os.listdir(entry.path):
                        if fname.lower().endswith(IMG_EXTS):
                            self.samples.append((os.path.join(entry.path, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError("No samples found.")

        print(f"[ImageNetLatentCacheDataset] samples={len(self.samples)} cache_root={self.cache_root}")

    def __len__(self):
        return len(self.samples)

    def cache_path_for(self, img_path: str) -> str:
        rel = sample_cache_relpath(img_path, self.tag)
        return os.path.join(self.cache_root, rel)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label = self.samples[idx]
        cache_path = self.cache_path_for(img_path)

        if os.path.exists(cache_path):
            obj = torch.load(cache_path, map_location="cpu")
            # obj["latent"] expected [C,H,W] tensor
            return {
                "has_latent": True,
                "latent": obj["latent"],
                "label": int(obj.get("label", label)),
                "path": img_path,
                "cache_path": cache_path,
            }

        # cache miss: read image tensor and return
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return {
            "has_latent": False,
            "img": img,
            "label": int(label),
            "path": img_path,
            "cache_path": cache_path,
        }


def collate_keep_dict(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 保持 dict 结构，main 里自己拆
    return {
        "has_latent": torch.tensor([b["has_latent"] for b in batch], dtype=torch.bool),
        "latent": [b.get("latent", None) for b in batch],
        "img": [b.get("img", None) for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "path": [b["path"] for b in batch],
        "cache_path": [b["cache_path"] for b in batch],
    }
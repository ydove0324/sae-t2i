import yaml
import os
from torch.utils.data import Dataset
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
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

import torch
# from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from modeling_dino_v3 import DINOv3ViTModel

from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

from common.fs import download, exists


def mask_channels(tensor, mask_ratio=0.1, channel_dim=1):
    """
    Randomly sets a percentage of channels in the input tensor to zero.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        mask_ratio (float): The fraction of channels to mask (0.0 to 1.0).
        channel_dim (int): The dimension representing the channels.
        
    Returns:
        torch.Tensor: A new tensor with masked channels.
    """
    # Clone to avoid modifying the original tensor in-place
    output = tensor.clone()
    
    # Get the total number of channels
    num_channels = tensor.shape[channel_dim]
    
    # Calculate how many channels to mask
    num_masked = int(num_channels * mask_ratio)
    
    if num_masked == 0:
        return output
    
    # Generate random permutation of channel indices and pick the first 'num_masked'
    mask_indices = torch.randperm(num_channels)[:num_masked]
    
    # Create a simplified index object to access the dynamic channel dimension
    # This creates a slice(None) [equivalent to :] for every dimension
    idx = [slice(None)] * tensor.ndim
    
    # Replace the slice for the channel dimension with our specific indices
    idx[channel_dim] = mask_indices
    
    # Set the selected channels to zero
    output[idx] = 0
    
    return output, mask_indices


if __name__ == "__main__":
    # unit test for using dino v3
    def get_img():
        import requests
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        return image

    def make_transform(resize_size: int | list[int] = 768):
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((resize_size, resize_size), antialias=True)
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([to_tensor, resize, normalize])

    # dinov3_hdfs_path = "hdfs://harunava/home/byte_data_seed_vagcp/user/ming.li1/vfm_exp/pretrained_models/dinov3-vith16plus-pretrain-lvd1689m/model.safetensors"
    dinov3_hdfs_path = "hdfs://harunawl/home/byte_data_seed_wl/vgfm/users/xiaojieli/dinov3/dinov3-vith16plus-pretrain-lvd1689m/model.safetensors"
    assert exists(dinov3_hdfs_path), "The dinov3 weights on HDFS is not found."
    download(
        dinov3_hdfs_path,
        dirname="/opt/tiger/vfm/models/dinov3-vith16plus-pretrain-lvd1689m/",
        filename="model.safetensors",
    )

    dinov3_config_hdfs_path = [
        # VA
        # "hdfs://harunava/home/byte_data_seed_vagcp/user/ming.li1/vfm_exp/pretrained_models/dinov3-vith16plus-pretrain-lvd1689m/config.json",
        # "hdfs://harunava/home/byte_data_seed_vagcp/user/ming.li1/vfm_exp/pretrained_models/dinov3-vith16plus-pretrain-lvd1689m/preprocessor_config.json",
        # CN
        "hdfs://harunawl/home/byte_data_seed_wl/vgfm/users/xiaojieli/dinov3/dinov3-vith16plus-pretrain-lvd1689m/config.json",
        "hdfs://harunawl/home/byte_data_seed_wl/vgfm/users/xiaojieli/dinov3/dinov3-vith16plus-pretrain-lvd1689m/preprocessor_config.json",
    ]
    for config_hdfs_path in dinov3_config_hdfs_path:
        assert exists(
            config_hdfs_path
        ), f"The dinov3 config on HDFS is not found: {config_hdfs_path}"
        download(
            config_hdfs_path,
            dirname="/opt/tiger/vfm/models/dinov3-vith16plus-pretrain-lvd1689m/",
            filename=f"{config_hdfs_path.split('/')[-1]}",
        )

    dinov3_vith16plus = DINOv3ViTModel.from_pretrained(
        pretrained_model_name_or_path='/opt/tiger/vfm/models/dinov3-vith16plus-pretrain-lvd1689m/', 
        use_safetensors=True
    )

    img_size = 256
    img  = get_img()
    transform = make_transform(img_size)
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            batch_img = transform(img)[None]
            pred_vit7b = dinov3_vith16plus(batch_img)  # raw predictions 

    print(pred_vit7b)
    print(f"pred_vit7b.last_hidden_state shape is {pred_vit7b.last_hidden_state.shape}")
    print(f"pred_vit7b.pooler_output shape is {pred_vit7b.pooler_output.shape}")

    # 假设 cls_token 原始形状为 (1, 1, embed_dim)
    # register_tokens 原始形状为 (1, num_register, embed_dim)

    # 从 embeddings 中恢复 patch_embeddings
    # 1. 确定 cls_token 和 register_tokens 的长度
    cls_len = 1  # cls_token 通常只有1个
    register_len = 4  # 注册token的数量

    # 2. 切片获取 patch_embeddings
    patch_embeddings_restored = pred_vit7b.last_hidden_state[:, cls_len + register_len :, :]

    restored_tensor = patch_embeddings_restored.transpose(1, 2).view(patch_embeddings_restored.shape[0], -1, 16, 16)

    print(f"restored_tensor shape is {restored_tensor.shape}")
    print(f"restored_tensor content is {restored_tensor}")

    # Apply the mask
    h_masked, indices_zeroed = mask_channels(restored_tensor, mask_ratio=0.1, channel_dim=1)

    # Verification
    print("-" * 30)
    print(f"Number of channels: {restored_tensor.shape[1]}")
    print(f"Number of channels masked (10%): {len(indices_zeroed)}")
    print(f"Indices set to zero (first 10 shown): {indices_zeroed[:10].tolist()}...")

     # Check if the specific channels are actually zero
    # We check the first sample in the batch, and the first pixel (0,0)
    is_zeroed = True
    for idx in indices_zeroed:
        # Select the channel 'idx' across all batches and spatial dimensions
        channel_data = h_masked[:, idx, :, :]
        # print(f"here is the channel data {channel_data}")
        if torch.count_nonzero(channel_data) > 0:
            is_zeroed = False
            break

    print(f"Verification - Are selected channels all zeros? {is_zeroed}")
    print(f"New Norm: {torch.norm(h_masked):.4f}")

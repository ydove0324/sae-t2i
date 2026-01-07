sudo apt-get install -y unzip tmux python3-venv
mkdir -p /opt/tiger/dataset
hdfs dfs -get hdfs://harunava/home/byte_data_seed_azureb/user/yancheng.zhang/imagenet/imagenet.zip /opt/tiger/dataset
unzip /opt/tiger/dataset/imagenet.zip -d /opt/tiger/dataset

# hdfs dfs -get hdfs://haruna/home/byte_arnold_lq/user/yibairen.byron/imagenet/ILSVRC2012_img_train_full.tar /opt/tiger/dataset
# tar xf /opt/tiger/dataset/ILSVRC2012_img_train_full.tar -C /opt/tiger/dataset

# vae without noise
# mkdir -p /opt/tiger/vfm/weights/dinov3
# hdfs dfs -get \
#     hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/dino_vae_e2e_init-stage2-630k_kl100_fix_11-11-val_imagenet/states/0000100000/models \
#     /opt/tiger/vfm/weights/dinov3
# hdfs dfs -get \
#     hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/dino_vae_e2e_init-stage2-630k_kl100_fix_11-11-val_imagenet/configs \
#     /opt/tiger/vfm/weights/dinov3

mkdir -p /opt/tiger/vfm/weights/dinov3
hdfs dfs -get \
    hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/1026_from_pt_masked_nosied_sae-e2e_0.1_0.8_denormalize/states/0000110000/models \
    /opt/tiger/vfm/weights/dinov3
hdfs dfs -get \
    hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/1026_from_pt_masked_nosied_sae-e2e_0.1_0.8_denormalize/configs \
    /opt/tiger/vfm/weights/dinov3

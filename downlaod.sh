# hdfs dfs -ls hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/dino_vae_e2e/states/0000417500/models

# 16x
# hdfs dfs -get \
#     hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/dino_vae_e2e/states/0000090000/models/ \
#     ./weights/dinov3_s16_kl0

# 32x
# hdfs dfs -get \
#     hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/dino_vae_decoder_only_32x/states/0000480000/models/ \
#     ./weights/dinov3_s32


# 64x
hdfs dfs -get \
    hdfs://haruna/home/byte_data_seed/hdd_hldy/user/ming.li1/work_dirs/vfm_exp/ming_dino_vae_exp/dino_vae_decoder_only_64x/states/0000480000/models/ \
    ./weights/dinov3_s64
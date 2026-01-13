export TORCH_HOME="/share/project/huangxu/.cache/torch"
python inference.py \
  --config /share/project/huangxu/SAE/projects/rae/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv3_1536.yaml \
  --ckpt   /share/project/huangxu/SAE/result_v4/checkpoints/0400000.pt \
  --vae-ckpt /share/project/huangxu/sae_hx/diff_decoder/frozen_enc_vae_76000.pth \
  --label all \
  --image-size 256 \
  --stage2-steps 50 \
  --batch-size 16 \
  --vae-diffusion-steps 50 \
  --out result/frozen_test \
  --seed 0 \
  --use-ema


python inference.py \
  --config /share/project/huangxu/SAE/projects/rae/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv3_1536.yaml \
  --ckpt   /share/project/huangxu/SAE/result_v3/checkpoints/0340000.pt \
  --vae_ckpt /share/project/huangxu/sae_hx/checkpoints_decoder_v6/decoder_step_2000.pth \
  --label all \
  --image-size 256 \
  --stage2-steps 50 \
  --batch-size 16 \
  --vae-diffusion-steps 50 \
  --out result/tuned_dinov3_test \
  --seed 0 \
  --use-ema

   # --vae-ckpt /share/project/huangxu/sae_hx/diff_decoder/tuned_enc_vae_26000.pth \
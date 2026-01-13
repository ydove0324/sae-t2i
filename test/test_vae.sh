python test/test_vae.py \
  --type CNN \
  --ckpt "/share/project/huangxu/models/SAE/models/ema_vae.pth" \
  --input "dinov3_overfit.png" \
  --output "test/recon_cnn.png"
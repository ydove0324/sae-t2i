torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  projects/rae/sample_ddp.py \
  --config /opt/tiger/vfm/projects/rae/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv3-1536.yaml \
  --sample-dir /opt/tiger/vfm/samples/test \
  --precision bf16 \
  --label-sampling equal \
  --per-proc-batch-size 128

# python /opt/tiger/vfm/projects/rae/guided-diffusion/evaluations/evaluator.py /opt/tiger/vfm/VIRTUAL_imagenet256_labeled.npz /opt/tiger/vfm/samples/DiTwDDTHead-latest-cfg-1.00-bs8-ODE-50-euler-bf16.npz
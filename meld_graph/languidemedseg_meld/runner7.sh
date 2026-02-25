while nvidia-smi -i 2 | grep -q "python"; do
  echo "GPU 2 busy, waiting..."
  sleep 1200
done

echo "GPU 2 free, starting job"
# CUDA_VISIBLE_DEVICES=1 python3 languidemedseg_meld/train_Kfold.py \
#   --config languidemedseg_meld/config/training.yaml \
#   --ckpt_path ./languidemedseg_meld/save_model/exp1_no_gnn_full_aug \
#   --job_name exp3_mixed_3_no_gnn_aug
CUDA_VISIBLE_DEVICES=2 python3 languidemedseg_meld/train_Kfold_copy.py --config languidemedseg_meld/config/training.yaml --job_name exp3_lobe_no_aug
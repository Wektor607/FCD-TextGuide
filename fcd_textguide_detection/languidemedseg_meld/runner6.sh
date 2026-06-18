while nvidia-smi -i 3 | grep -q "python"; do
  echo "GPU 3 busy, waiting..."
  sleep 1200
done

echo "GPU 3 free, starting job"
# CUDA_VISIBLE_DEVICES=3 python3 languidemedseg_meld/test_Kfold.py --config languidemedseg_meld/config/training.yaml --ckpt_prefix save_model/exp3_no_gnn_lobe_general_freeze
# CUDA_VISIBLE_DEVICES=3 python3 languidemedseg_meld/train_Kfold.py \
#   --config languidemedseg_meld/config/training.yaml \
#   --ckpt_path ./languidemedseg_meld/save_model/exp1_no_gnn_full_aug \
#   --job_name exp3_mixed_3_no_gnn_aug
CUDA_VISIBLE_DEVICES=3 python3 languidemedseg_meld/train_Kfold.py --config languidemedseg_meld/config/training.yaml --job_name exp3_hemi_no_gnn_aug


# CUDA_VISIBLE_DEVICES=1 python3 languidemedseg_meld/train_Kfold.py --config languidemedseg_meld/config/training.yaml --job_name exp3_mixed_no_gnn --ckpt_path /raid/Users/mikhelson/FCD-Detection/meld_graph/save_model/exp1_no_gnn_aug
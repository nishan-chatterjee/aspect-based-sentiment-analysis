# For Slovenian
export CUDA_VISIBLE_DEVICES=4 && python3 "6.4 hierarchical-attention-networks.py" \
  --split slovenian \
  --method_name global-context-modelling/with-aspect-markers \
  --model_name xlm-roberta-base \
  --epochs 10 \
  --batch_size 4 \
  --eff_batch_size_target 32 \
  --lr 1e-5 \
  --max_sentences 128 \
  --max_seq_length 96 \
  --interaction_layers 2 \
  --interaction_heads 8 \
  --aggregation_heads 4 \
  --dropout_rate 0.2 \
  --final_mlp_hidden_dim 256 \
  --use_aspect_marker

# For Serbian
export CUDA_VISIBLE_DEVICES=4 && python3 "6.4 hierarchical-attention-networks.py" \
  --split serbian \
  --method_name global-context-modelling/with-aspect-markers \
  --model_name xlm-roberta-base \
  --epochs 10 \
  --batch_size 4 \
  --eff_batch_size_target 32 \
  --lr 1e-5 \
  --max_sentences 128 \
  --max_seq_length 96 \
  --interaction_layers 2 \
  --interaction_heads 8 \
  --aggregation_heads 4 \
  --dropout_rate 0.2 \
  --final_mlp_hidden_dim 256 \
  --use_aspect_marker
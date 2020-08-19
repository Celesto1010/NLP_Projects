export DATA_DIR=./data
export MODEL_DIR=/alidata1/wyd/NLP/Models/GPT2/Poets_Five
export TOKENIZER_VOCAB=./vocab/vocab_small.txt
export BERT_TOKENIZER_DIR=/alidata1/wyd/NLP/Bert_Pretrained/Chinese/Bert_Tokenizer

python ./run_gpt2.py \
  --mode train  \
  --train true \
  --eval true \
  --train_batch_size 16 \
  --num_epoch 10 \
  --stride 768 \
  --min_length 0 \
  --lr 1e-5 \
  --warmup_step 1000  \
  --max_grad_norm 1.0 \
  --data_dir $DATA_DIR \
  --output_dir $MODEL_DIR
  --tokenizer_vocab $TOKENIZER_VOCAB
  --bert_tokenizer_dir $BERT_TOKENIZER_DIR
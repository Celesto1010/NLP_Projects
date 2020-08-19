export BERT_EN_DIR=/alidata1/wyd/NLP/Bert_Pretrained/English_uncased
export BERT_ZH_DIR=/alidata1/wyd/NLP/Bert_Pretrained/Chinese
export MODEL_DIR=/alidata1/wyd/NLP/Models/Translation/Bert
export DATA_DIR=/alidata1/wyd/NLP/Datasets/Translation_ZH-EN_wmt19

python ./run_translator.py \
  --mode train  \
  --train true \
  --eval true \
  --num_epoch 60 \
  --train_batch_size 20 \
  --lr 1e-5 \
  --warmup_step 50000  \
  --tokenizer_en_dir $BERT_EN_DIR/Bert_Tokenizer
  --tokenizer_zh_dir $BERT_ZH_DIR/Bert_Tokenizer
  --model_config_dir ./ed_config \
  --pretrained_english_dir $BERT_EN_DIR/Bert_Pretrained_Models/BertModel \
  --pretrained_chinese_dir $BERT_ZH_DIR/Bert_Pretrained_Models/BertModel \
  --model_dir $MODEL_DIR \
  --data_dir $DATA_DIR
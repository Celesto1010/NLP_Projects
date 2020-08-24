export TOKENIZER_VOCAB=./vocab/vocab_small.txt
export MODEL_PATH=/alidata1/wyd/NLP/Models/GPT2/Poets_Five


python ./generate.py \
  --length 100 \
  --nsamples 5 \
  --tokenizer_path $TOKENIZER_VOCAB \
  --model_path $MODEL_PATH \
  --model_dir model_epoch_15 \
  --prefix 床前明月光
## 机器翻译

### Dir Transformer  
基于pytorch，搭建transformer的完整encoder-decoder模型。未使用bert tokenizer  
-- build_wmt19_data: wmt19翻译语料数据清洗  
-- data_pretreat: 另一个语料的数据清洗  
-- datasets: 数据预处理，生成pytorch的Datasets类  
-- models: Transformer模型结构  
-- run_translator: 运行模型训练/测试  
-- tokenization: 分词器  

### Bert-Base_Transformer  
使用经过预训练的bert模型搭建transformer模型。encoder使用英文预训练模型，decoder使用中文预训练模型  
-- datasets: 数据预处理  
-- run_translation: 得到tokenizer、model结构，运行模型训练/测试  

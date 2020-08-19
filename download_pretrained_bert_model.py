import os
import transformers


BASE_BERT_MODEL_DIR = r'E:\NLP_Projects\Models\Bert_Pretrained'
if not os.path.exists(BASE_BERT_MODEL_DIR):
    os.mkdir(BASE_BERT_MODEL_DIR)


def download_model():
    """ 下载中文用与英文用的预训练模型
        分为3类，预训练阶段模型只下载ForMaskedLM，FineTune预训练模型不提前下载
            1. 基本款：
                BertModel
                BertTokenizer
            2. 预训练阶段：
                BertForMaskedLM
                BertForNextSequencePrediction
                BertForPreTraining
            3. Fine-tune阶段：
                BertForSequenceClassification
                BertForTokenClassification
                BertForQuestionAnswering
                BertForMultipleChoice"""
    english_dir = os.path.join(BASE_BERT_MODEL_DIR, 'English_uncased')
    chinese_dir = os.path.join(BASE_BERT_MODEL_DIR, 'Chinese')
    if not os.path.exists(english_dir):
        os.mkdir(english_dir)
    if not os.path.exists(chinese_dir):
        os.mkdir(chinese_dir)

    # 1. 下载tokenizer
    print('正在下载tokenizer')
    en_tokenizer_dir = os.path.join(english_dir, 'Bert_Tokenizer')
    if os.path.exists(en_tokenizer_dir):
        print('英文tokenizer已下载')
    else:
        os.mkdir(en_tokenizer_dir)
        en_tokenizer_cache_dir = os.path.join(en_tokenizer_dir, 'cache')
        os.mkdir(en_tokenizer_cache_dir)
        en_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=en_tokenizer_cache_dir)
        en_tokenizer.save_pretrained(en_tokenizer_dir)

    zh_tokenizer_dir = os.path.join(chinese_dir, 'Bert_Tokenizer')
    if os.path.exists(zh_tokenizer_dir):
        print('中文tokenizer已下载')
    else:
        os.mkdir(zh_tokenizer_dir)
        zh_tokenizer_cache_dir = os.path.join(zh_tokenizer_dir, 'cache')
        os.mkdir(zh_tokenizer_cache_dir)
        zh_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=zh_tokenizer_cache_dir)
        zh_tokenizer.save_pretrained(zh_tokenizer_dir)
    print('tokenizer下载完毕')
    print('-' * 50)

    # 2. 下载预训练模型
    print('正在下载预训练模型')
    en_model_dir = os.path.join(english_dir, 'Bert_Pretrained_Models')
    if not os.path.exists(en_model_dir):
        os.mkdir(en_model_dir)
    zh_model_dir = os.path.join(chinese_dir, 'Bert_Pretrained_Models')
    if not os.path.exists(zh_model_dir):
        os.mkdir(zh_model_dir)
    # 2.1 下载BertModel
    print('正在下载BertModel')
    en_bertmodel_dir = os.path.join(en_model_dir, 'BertModel')
    if os.path.exists(en_bertmodel_dir):
        print('英文BertModel已下载')
    else:
        os.mkdir(en_bertmodel_dir)
        en_bertmodel_cache_dir = os.path.join(en_bertmodel_dir, 'cache')
        os.mkdir(en_bertmodel_cache_dir)
        en_bertmodel = transformers.BertModel.from_pretrained('bert-base-uncased', cache_dir=en_bertmodel_cache_dir)
        en_bertmodel.save_pretrained(en_bertmodel_dir)

    zh_bertmodel_dir = os.path.join(zh_model_dir, 'BertModel')
    if os.path.exists(zh_bertmodel_dir):
        print('中文BertModel已下载')
    else:
        os.mkdir(zh_bertmodel_dir)
        zh_bertmodel_cache_dir = os.path.join(zh_bertmodel_dir, 'cache')
        os.mkdir(zh_bertmodel_cache_dir)
        zh_bertmodel = transformers.BertModel.from_pretrained('bert-base-chinese', cache_dir=zh_bertmodel_cache_dir)
        zh_bertmodel.save_pretrained(zh_bertmodel_dir)
    print('BertModel下载完毕')
    print('-' * 20)

    # 2.2 下载ForMaskedLM
    print('正在下载BertForMaskedLM')
    en_bertformaskedlm_dir = os.path.join(en_model_dir, 'BertForMaskedLM')
    if os.path.exists(en_bertformaskedlm_dir):
        print('英文BertModelLM已下载')
    else:
        os.mkdir(en_bertformaskedlm_dir)
        en_bertformaskedlm_cache_dir = os.path.join(en_bertformaskedlm_dir, 'cache')
        os.mkdir(en_bertformaskedlm_cache_dir)
        en_bertformaskedlm = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased',
                                                                          cache_dir=en_bertformaskedlm_cache_dir)
        en_bertformaskedlm.save_pretrained(en_bertformaskedlm_dir)

    zh_bertformaskedlm_dir = os.path.join(zh_model_dir, 'BertForMaskedLM')
    if os.path.exists(zh_bertformaskedlm_dir):
        print('中文BertModelLM已下载')
    else:
        os.mkdir(zh_bertformaskedlm_dir)
        zh_bertformaskedlm_cache_dir = os.path.join(zh_bertformaskedlm_dir, 'cache')
        os.mkdir(zh_bertformaskedlm_cache_dir)
        zh_bertformaskedlm = transformers.BertForMaskedLM.from_pretrained('bert-base-chinese',
                                                                          cache_dir=zh_bertformaskedlm_cache_dir)
        zh_bertformaskedlm.save_pretrained(zh_bertformaskedlm_dir)
    print('BertModel下载完毕')
    print('-' * 20)


if __name__ == '__main__':
    download_model()


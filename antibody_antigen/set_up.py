import torch
from tape import TAPETokenizer, ProteinBertModel

BERT_PATH = "/antibody_antigen_model/pretrain_bert.models"
model = ProteinBertModel.from_pretrained('bert-base')
torch.save(model, BERT_PATH)
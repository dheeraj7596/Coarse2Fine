import torch
from transformers import BertModel
from contrastive_vmf import contrastiveNLLvMF


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 500)
        self.l4 = torch.nn.Linear(500, 300)
        self.l5 = torch.nn.Linear(300, 100)

    def forward(self, ids, label_embeddings, attention_mask, token_type_ids, labels, device):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output = self.l5(output_4)
        loss, logits = contrastiveNLLvMF(output, labels, label_embeddings, device)
        return loss, logits

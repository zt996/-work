import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score



# 模型路径和数据路径
bert_path = r"C:\Users\web\Desktop\bert-base-uncased"
glue_path = r"C:\Users\web\Desktop\glue"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(bert_path)
model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=2)
model.to(device)

# 数据集定义（支持单句和句对任务）
class GLUEDataset(Dataset):
    def __init__(self, df, tokenizer, sentence1_key, sentence2_key=None, max_length=128):
        self.sentence1 = df[sentence1_key].tolist()
        self.sentence2 = df[sentence2_key].tolist() if sentence2_key else None
        self.labels = df['Quality'].astype(int).tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        args = {
            "truncation": True,
            "padding": 'max_length',
            "max_length": self.max_length,
            "return_tensors": 'pt'
        }
        if self.sentence2:
            encoding = self.tokenizer(self.sentence1[idx], self.sentence2[idx], **args)
        else:
            encoding = self.tokenizer(self.sentence1[idx], **args)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# 训练 + 验证函数
def train_eval(train_dataset, dev_dataset, name):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"[MRPC] 训练中，第 {step + 1}/{len(train_loader)} 个 batch，loss={loss.item():.4f}")

    # 评估
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=-1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    print(f"{name} 任务：准确率 = {acc:.4f}，F1 = {f1:.4f}")

# ----------- 任务2：MRPC（句对语义相似性）-----------
print("任务2：MRPC")
mrpc_path = os.path.join(glue_path, 'MRPC')
mrpc_df = pd.read_csv(os.path.join(mrpc_path, 'MRPC.tsv'), sep='\t')
mrpc_train = mrpc_df.sample(frac=0.8, random_state=42)
mrpc_dev = mrpc_df.drop(mrpc_train.index)


mrpc_train_set = GLUEDataset(mrpc_train, tokenizer, sentence1_key='#1 String', sentence2_key='#2 String')
mrpc_dev_set = GLUEDataset(mrpc_dev, tokenizer, sentence1_key='#1 String', sentence2_key='#2 String')

train_eval(mrpc_train_set, mrpc_dev_set, name='MRPC')

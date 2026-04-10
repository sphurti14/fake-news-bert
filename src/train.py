import torch

def encode_data(df, tokenizer):
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    labels = torch.tensor(labels)

    return input_ids, attention_mask, labels

from torch.utils.data import DataLoader, TensorDataset

def train_model(model, input_ids, attention_mask, labels):
    dataset = TensorDataset(input_ids, attention_mask, labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()

    for epoch in range(1):  # 1 epoch for now
        for batch in loader:
            input_ids, attention_mask, labels = batch

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("Loss:", loss.item())
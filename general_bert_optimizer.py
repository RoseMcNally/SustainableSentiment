import numpy as np
from transformers import AutoModel, BertTokenizerFast, AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import AdamW
from general_bert_arch import GeneralBertArch


class BertModelOptimizer:
    train_data = None
    train_sampler = None
    val_data = None
    val_sampler = None
    run_count = 0
    best_model_parameters = None

    def __init__(self, model_name, device, best_model_path, training_epochs=10):
        self.bert = AutoModel.from_pretrained(model_name)
        self.device = device
        self.epochs = training_epochs
        self.best_model_path = best_model_path
        self.best_loss = float('inf')

    def load_training_data(self, seq, mask, y):
        self.train_data = TensorDataset(seq, mask, y)
        self.train_sampler = RandomSampler(self.train_data)

    def load_validation_data(self, seq, mask, y):
        self.val_data = TensorDataset(seq, mask, y)
        self.val_sampler = SequentialSampler(self.val_data)

    def run(self, batch_size, learning_rate, weight_decay):

        if any([self.train_data, self.train_sampler, self.val_data, self.val_sampler]) is None:
            print("You need to load the data before running the optimizer!")
            return

        train_dataloader = DataLoader(self.train_data, sampler=self.train_sampler, batch_size=batch_size)
        val_dataloader = DataLoader(self.val_data, sampler=self.val_sampler, batch_size=batch_size)

        model = GeneralBertArch(self.bert)
        model = model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        class_weights = [1, 1]
        weights = torch.tensor(class_weights, dtype=torch.float)
        weights = weights.to(self.device)

        cross_entropy = nn.NLLLoss(weight=weights)

        self.run_count += 1
        print("Run", self.run_count, "- Batch size:", batch_size, ", Learning rate:", learning_rate, ", Weight decay:",
              weight_decay)

        for epoch in range(self.epochs):

            print('Epoch {:} / {:}'.format(epoch + 1, self.epochs))

            train_loss, _ = self.train(model, train_dataloader, optimizer, cross_entropy)

            valid_loss, _ = self.evaluate(model, val_dataloader, cross_entropy)

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(model.state_dict(), self.best_model_path + '/best_weights.pt')
                self.best_model_parameters = (batch_size, learning_rate, weight_decay)

            print(f'\tTraining Loss: {train_loss:.3f}')
            print(f'\tValidation Loss: {valid_loss:.3f}')

    def best_model(self):
        model = GeneralBertArch(self.bert)
        model = model.to(self.device)
        model.load_state_dict(torch.load(self.best_model_path + "/best_weights.pt"))
        return model

    def train(self, model, train_dataloader, optimizer, cross_entropy):

        model.train()

        total_loss, total_accuracy = 0, 0
        total_preds = []

        for step, batch in enumerate(train_dataloader):
            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch

            model.zero_grad()

            preds = model(sent_id, mask)

            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

        avg_loss = total_loss / len(train_dataloader)

        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def evaluate(self, model, val_dataloader, cross_entropy):

        model.eval()

        total_loss, total_accuracy = 0, 0
        total_preds = []

        for step, batch in enumerate(val_dataloader):
            batch = [t.to(self.device) for t in batch]

            sent_id, mask, labels = batch

            with torch.no_grad():
                preds = model(sent_id, mask)

                loss = cross_entropy(preds, labels)

                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

        avg_loss = total_loss / len(val_dataloader)

        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds


class BertTokenizer:

    def __init__(self, tokenizer_name, max_len):
        if tokenizer_name == "vinai/bertweet-base":
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)
        elif tokenizer_name == "bert-base-uncased":
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        else:
            print("Invalid tokenizer name:", tokenizer_name)
            self.tokenizer = None
        self.max_len = max_len

    def tokenize(self, text_data, labels):
        tokens = self.tokenizer.batch_encode_plus(
            text_data.tolist(),
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        seq = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])
        y = torch.tensor(labels.tolist())

        return seq, mask, y

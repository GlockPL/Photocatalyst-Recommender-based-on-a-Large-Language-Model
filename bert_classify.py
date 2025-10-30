import numpy as np
import random
from tqdm import tqdm
import pickle
import re
from sklearn.model_selection import StratifiedKFold

import torch
from torch.optim import AdamW
from transformers import RobertaTokenizer
from torchmetrics import Accuracy, F1Score
from torch import nn

from transformers import AutoModelForSequenceClassification


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CatalsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return self.encodings[idx]

    def __len__(self):
        return len(self.encodings)




def main():
    # Path to pretrained model
    model_path = "./pretrained_reaction_count_3119043_word_len_1024"
    max_len_re = re.findall(r'word_len_([0-9]{3,4})', model_path)
    max_len = int(max_len_re[0])
    # Path to dict with the finetunning dataset
    with open('./data/bertaction_corrected_catals.pickle', 'rb') as pickle_file:
        data_dict = pickle.load(pickle_file)
    first_class = np.argmax(np.array(data_dict['y'][0]))
    print(
        f"First React: {data_dict['reactions'][0]} and Group: {data_dict['groups'][0]}. Corresponding class: {first_class}")

    print("Tokenizing!")
    print(f"Word length:{Bcolors.OKBLUE} {max_len}{Bcolors.ENDC}")
    # 512 514
    tokenizer = RobertaTokenizer.from_pretrained("./BPETokenizer/")
    batch_token = tokenizer(data_dict['reactions'], data_dict['groups'], max_length=max_len, padding='max_length',
                            truncation=True, return_tensors='pt')
    print("Tokenization finished!")

    print(f"First Tokenized seq: {batch_token.input_ids[0, :]}")
    y = np.array(data_dict["y"])
    class_amount = y.shape[1]
    y = np.argmax(y, axis=1)
    y_class = y

    attention_mask = batch_token.attention_mask
    labels = torch.tensor(y)

    rng = labels.shape[0]
    dataset = []

    for i in range(rng):
        row = {"input_ids": batch_token.input_ids[i, :], "attention_mask": attention_mask[i, :], "labels": labels[i]}
        dataset.append(row)
    print(f"{Bcolors.OKBLUE}Size of the dataset: {len(dataset)}{Bcolors.ENDC}")
    random.shuffle(dataset)
    n_splits = 5
    accs = []
    f1s = []
    f1s_macro = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=43)

    fold = 0
    fold_indices = {}
    for train_index, test_index in skf.split(dataset, y_class):
        model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                   num_labels=class_amount,
                                                                   id2label=data_dict['catal_num_to_smiles_map'],
                                                                   label2id=data_dict['catal_smiles_to_num_map'])
        fold += 1
        print(f"Current fold: {fold}")

        dataset_train = [dataset[x] for x in train_index]
        dataset_test = [dataset[x] for x in test_index]
        print(f"{Bcolors.OKBLUE}Training size: {len(dataset_train)}, valid size: {len(dataset_test)}{Bcolors.ENDC}")
        dataloader_train = torch.utils.data.DataLoader(CatalsDataset(dataset_train), batch_size=16)
        dataloader_test = torch.utils.data.DataLoader(CatalsDataset(dataset_test), batch_size=6)

        print(
            f"{Bcolors.OKBLUE}Training batched size: {len(dataloader_train)}, valid batched size: {len(dataloader_test)}{Bcolors.ENDC}")

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = nn.DataParallel(model)
        model.to(device)
        model.train()

        train_loader = dataloader_train

        optim = AdamW(model.parameters(), lr=2e-5)
        confmat = Accuracy(num_classes=class_amount, task="multiclass").to(device)
        f1 = F1Score(num_classes=class_amount, average='weighted', task="multiclass").to(device)
        f1_macro = F1Score(num_classes=class_amount, average='macro', task="multiclass").to(device)
        epochs = 6
        min_valid_loss = np.inf
        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            loop = tqdm(train_loader, leave=True)
            train_acc = 0.0
            train_f1 = 0.0
            train_f1_macro = 0.0

            for batch in loop:
                optim.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()
                optim.step()
                train_loss += loss.item()
                current_acc = confmat(logits, labels)
                train_acc += current_acc
                train_f1 += f1(logits, labels)
                train_f1_macro += f1_macro(logits, labels)
                loop.set_description(f'Epoch {epoch + 1}')
                loop.set_postfix(loss=loss.item(), acc=current_acc.detach().cpu().numpy())

            print(
                f"{Bcolors.OKBLUE}Training accuracy: {(train_acc / len(train_loader)):.4f}, Training F1: {(train_f1 / len(train_loader)):.4f}, Training F1 Macro: {(train_f1_macro / len(train_loader)):.4f}{Bcolors.ENDC}")

            valid_loss = 0.0
            acc = 0.0
            val_f1 = 0.0
            val_f1_macro = 0.0
            model.eval()
            for batch in tqdm(dataloader_test, leave=True):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                labels = batch["labels"]
                acc += confmat(logits, labels)
                val_f1 += f1(logits, labels)
                val_f1_macro += f1_macro(logits, labels)
                valid_loss += loss.item()

            if epoch == epochs - 1:
                accs.append(acc.cpu() / len(dataloader_test))
                f1s.append(val_f1.cpu() / len(dataloader_test))
                f1s_macro.append(val_f1_macro.cpu() / len(dataloader_test))

            print(
                f'Epoch {epoch + 1} \t\t Training Loss: {(train_loss / len(train_loader)):.4f} \t Validation Loss: {(valid_loss / len(dataloader_test)):.4f} \t Validation Acc: {(acc / len(dataloader_test)):.4f} \t Validation F1: {(val_f1 / len(dataloader_test)):.4f}')
            print(
                f'Epoch {epoch + 1} \t\t Training F1 Macro: {(train_f1_macro / len(train_loader)):.4f} \t Validation F1 Macro: {(val_f1_macro / len(dataloader_test)):.4f}')
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')  # \t Saving The Model
                min_valid_loss = valid_loss

        model.eval()

        save_model = model.module
        # save_pretrained
        path_to_save = f'./KFold/reactioberto_classify_photocatals_fold_{fold}_val_acc_{(acc / len(dataloader_test)):.4f}_val_f1_{(val_f1 / len(dataloader_test)):.4f}'
        save_model.save_pretrained(path_to_save)
        del model

        print("Testing Model on training data for bugs!")

        model = AutoModelForSequenceClassification.from_pretrained(path_to_save)
        model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        acc = 0.0
        for batch in tqdm(train_loader, leave=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                prediction = model(**batch)
            logits = prediction.logits
            labels = batch["labels"]
            acc += confmat(logits, labels)

        print(f"Testing accuracy: {acc / len(train_loader)}")

        del model

        # Store indices for this fold
        fold_indices[f'fold_{fold}'] = {
            'train_indices': train_index.tolist(),
            'test_indices': test_index.tolist()
        }

    # Save fold indices to pickle file
    with open('data/fold_split_indices.pickle', 'wb') as f:
        pickle.dump(fold_indices, f)
    print(f"Fold indices saved to ./fold_split_indices.pickle")

    accs = np.array(accs)
    f1s = np.array(f1s)
    f1s_macro = np.array(f1s_macro)

    print(f"Mean val accuracy: {accs.mean()}, mean val f1: {f1s.mean()}, mean val f1 macro {f1s_macro.mean()}")
    res = ""
    for i in range(accs.shape[0]):
        res += f"Acc: {accs[i]}, F1: {f1s[i]}, F1 Macro: {f1s_macro[i]}\n"
    res += f"Acc Mean: {accs.mean()}, F1 Mean: {f1s.mean()}, F1 Mean Macro: {f1s_macro.mean()}"

    print(res)

    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=class_amount,
                                                               id2label=data_dict['catal_num_to_smiles_map'],
                                                               label2id=data_dict['catal_smiles_to_num_map'])
    dataloader_train = torch.utils.data.DataLoader(CatalsDataset(dataset), batch_size=16)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = nn.DataParallel(model)
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=2e-5)
    confmat_final = Accuracy(num_classes=class_amount, task="multiclass").to(device)
    epochs = 6
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        loop = tqdm(dataloader_train, leave=True)
        train_acc = 0.0

        for batch in loop:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optim.step()
            train_loss += loss.item()
            current_acc = confmat_final(logits, labels)
            train_acc += current_acc
            loop.set_description(f'Epoch {epoch + 1}')
            loop.set_postfix(loss=loss.item(), acc=current_acc.detach().cpu().numpy())

    model.eval()

    save_model = model.module

    save_model.save_pretrained(f'./reactioberto_classify_photocatals_corrected_dataset_word_len_{max_len}')


if __name__ == "__main__":
    main()

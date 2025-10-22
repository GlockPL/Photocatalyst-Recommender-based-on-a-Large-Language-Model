from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import pickle

unk_token = "<UNK>"  # token for unknown words
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens


def prepare_tokenizer_trainer():
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(special_tokens=spl_tokens)

    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer


def train_tokenizer(files):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer()
    tokenizer.train(files, trainer)  # training the tokenzier
    tokenizer.save("./Tokenizer/tokenizer-trained.json")
    tokenizer = Tokenizer.from_file("./Tokenizer/tokenizer-trained.json")
    return tokenizer


if __name__ == "__main__":
    with open('./pretraining_data.pickle', 'rb') as pickle_file:
        data_dict = pickle.load(pickle_file)

    lines = ""
    for react, group in zip(data_dict['reactions'],data_dict['groups']):
        lines += f"{react} {group} \n"

    file = "./reactions_groups.txt"
    with open(file, "w") as text_file:
        text_file.write(lines)

    train_tokenizer([file])


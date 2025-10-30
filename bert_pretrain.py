from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer


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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


def main():
    with open('data/pretraining_data.pickle', 'rb') as pickle_file:
        data_dict = pickle.load(pickle_file)
    print("Tokenizing...")
    # ========== CONFIG
    max_length = 1024
    epochs = 3
    batch_size = 32
    tokenizer = RobertaTokenizer.from_pretrained("./BPETokenizer/")
    samples = len(data_dict['reactions'])
    batch_token = tokenizer(data_dict['reactions'][:samples], data_dict['groups'][:samples], max_length=max_length, padding='max_length',
                            truncation=True, return_tensors='pt')
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,  # we align this to the tokenizer vocab_size
        max_position_embeddings=max_length+2,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1
    )
    print("Tokenization finished!")
    print(f"Training on: {Bcolors.OKBLUE} {len(data_dict['reactions'])} reactions {Bcolors.ENDC}")
    model = RobertaForMaskedLM(config)

    labels = batch_token.input_ids
    attention_mask = batch_token.attention_mask

    # make copy of labels tensor, this will be input_ids
    input_ids = labels.detach().clone()
    # create random array of floats with equal dims to input_ids
    rand = torch.rand(input_ids.shape)
    # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    # loop through each row in input_ids tensor (cannot do in parallel)
    for i in range(input_ids.shape[0]):
        # get indices of mask positions from mask array
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        # mask input_ids
        input_ids[i, selection] = 3  # our custom [MASK] token == 3

    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = nn.DataParallel(model)
    model.to(device)
    model.train()
    # initialize optimizer
    optim = AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for idx, batch in enumerate(loop):
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss.sum()
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    save_model = model
    save_model.save_pretrained(f'./pretrained_reaction_count_{len(data_dict["reactions"])}_word_len_{max_length}')

if __name__ == "__main__":
    main()


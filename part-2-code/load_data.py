import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.decoder_bos_id = self.tokenizer.pad_token_id
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        data_samples = []

        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)

            for nl, sql in zip(nl_lines, sql_lines):
                nl_with_prefix = f"translate English to SQL: {nl}"
                encoder_input = tokenizer(nl_with_prefix, return_tensors='pt', add_special_tokens=True)
                # encoder_input = tokenizer(nl, return_tensors='pt', add_special_tokens=True)
                encoder_ids = encoder_input['input_ids'].squeeze(0)
                encoder_mask = encoder_input['attention_mask'].squeeze(0)

                decoder_output = tokenizer(sql, return_tensors='pt', add_special_tokens=True)
                decoder_ids = decoder_output['input_ids'].squeeze(0)
                decoder_ids = torch.cat(
                    [
                        torch.tensor([self.decoder_bos_id], dtype=torch.long),
                        decoder_ids,
                    ],
                    dim=0,
                )

                data_samples.append({
                    'encoder_ids': encoder_ids,
                    'encoder_mask': encoder_mask,
                    'decoder_ids': decoder_ids,
                    'nl': nl,
                    'sql': sql
                })
        else:
            for nl in nl_lines:
                nl_with_prefix = f"translate English to SQL: {nl}"
                encoder_input = tokenizer(nl_with_prefix, return_tensors='pt', add_special_tokens=True)
                # encoder_input = tokenizer(nl, return_tensors='pt', add_special_tokens=True)
                encoder_ids = encoder_input['input_ids'].squeeze(0)
                encoder_mask = encoder_input['attention_mask'].squeeze(0)

                data_samples.append({
                    'encoder_ids': encoder_ids,
                    'encoder_mask': encoder_mask,
                    'nl': nl
                })

        return data_samples
    
    def __len__(self):
        # TODO
        return len(self.data)

    def __getitem__(self, idx):
        # TODO
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    encoder_mask_list = [item['encoder_mask'] for item in batch]
    decoder_ids_list = [item['decoder_ids'] for item in batch]

    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)
    decoder_full = pad_sequence(decoder_ids_list, batch_first=True, padding_value=PAD_IDX)

    decoder_inputs = decoder_full[:, :-1]
    decoder_targets = decoder_full[:, 1:]

    initial_decoder_inputs = decoder_full[:, 0:1]

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs
    # return [], [], [], [], []

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids_list = [item['encoder_ids'] for item in batch]
    encoder_mask_list = [item['encoder_mask'] for item in batch]

    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)

    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    initial_decoder_inputs = torch.full((len(batch), 1), tokenizer.pad_token_id, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs
    # return [], [], []

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x
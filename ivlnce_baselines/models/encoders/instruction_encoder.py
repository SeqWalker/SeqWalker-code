import gzip
import json

from nltk import sent_tokenize
from tensorflow.python.eager.context import device

from habitat.datasets.utils import VocabDict
import torch
import torch.nn as nn
from habitat import Config
from habitat.core.simulator import Observations
from torch import Tensor
from typing import List
import clip
import nltk
from nltk.tokenize import sent_tokenize
import math
import numpy as np
class InstructionEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config
        self.device = torch.device("cuda")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.instruction_vocab = None
        with open('my_data.json', 'r') as file:
            m = json.load(file)
        self.instruction_vocab = VocabDict(
            word_list=m
        )
        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )

        if config.sensor_uuid == "instruction":
            if self.config.use_pretrained_embeddings:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=not self.config.fine_tune_embeddings,
                )
            else:
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.embedding_size,
                    padding_idx=0,
                )

    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))

    def _load_embeddings(self) -> Tensor:

        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations: Observations) -> Tensor:


        m = observations
        p = 0
        instruction_vocab = self.instruction_vocab
        model = self.model
        if self.config.sensor_uuid == "instruction":
            instruction = observations["instruction"].long()
            select_instruction = []

            tensor_to_instruction = encoded_to_string(instruction,
                                                      instruction_vocab)

            instruction_split = process_text(tensor_to_instruction)

            rgb_image = observations["rgb"].permute(0, 3, 1, 2)
            for i in range(len(instruction_split)):
                p += 1
                instruction_split_pair = instruction_split[i]
                rgb_image_pair = rgb_image[i]
                select_instruction_pair = [
                    select_match_instruct(instruction_split_pair,
                                          rgb_image_pair, model,p)]
                select_instruction.append(select_instruction_pair)

            instruction = [instr[0] for instr in select_instruction]
            instruction = encode_instructions(instruction, instruction_vocab,
                                              max_length=200)
            lengths = (instruction != 0.0).long().sum(dim=1)
            instruction = self.embedding_layer(instruction)

        else:
            instruction = observations["rxr_instruction"]

        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu()

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        if self.config.rnn_type == "LSTM":
            final_state = final_state[0]

        if self.config.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1)

def truncate_text(text, max_length=77):
    words = text.split()
    if len(words) > max_length:
        words = words[:max_length]
    return " ".join(words)

def encoded_to_string(encoded_list, vocab):

    instructions = []
    for encoded_seq in encoded_list:
        words = [vocab.idx2word(idx.item()) for idx in encoded_seq if
                 idx.item() != 0]

        instruction = " ".join(words)
        instructions.append(instruction)
    return instructions


def select_match_instruct(split_text, image, model,p):
    device = "cuda"
    if split_text == ['<unk> '*199+'<unk>']:
        return '0'
    image = image.unsqueeze(0).to(device)
    text_tokens = clip.tokenize(split_text).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    ent = entropy(probs)
    if ent < 0.65:
        best_idx = probs.argmax()
        best_instruction = split_text[best_idx]
        return best_instruction
    else:
        return "".join(split_text)


def entropy(probs):
    probs = np.array(probs)
    valid_indices = probs > 0
    valid_probs = probs[valid_indices]
    ent = -np.sum(valid_probs * np.log2(valid_probs))

    return ent

def process_text(text_list):

    processed_texts = []
    for text in text_list:
        split_text = sent_tokenize(text)
        processed_texts.append(split_text)
    return processed_texts


def tokenize(sentence: str) -> List[str]:

    return sentence.lower().split()


def encode_instructions(instructions: List[str], vocab: VocabDict,
                        max_length: int = 200) -> torch.Tensor:

    encoded_sequences = []
    for instruction in instructions:
        words = tokenize(instruction)
        encoded = [vocab.word2idx(word) for word in words]
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        encoded += [vocab.PAD_INDEX] * (max_length - len(encoded))
        encoded_sequences.append(encoded)
    return torch.tensor(encoded_sequences, device="cuda")

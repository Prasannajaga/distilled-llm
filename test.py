from utils.binary_dataset import TokenizedBinaryDataset
from Cdatasets.tokenizer import load_tokenizer
tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

dataset = TokenizedBinaryDataset(
    source="/home/prasanna/.cache/huggingface/datasets/generator/default-caaf09e7aefe9c5f/0.0.0/",           
    tokenizer=tokenizer,
    block_size=512,
    bin_path="./data/automathtext.bin",  
    num_workers=8
)

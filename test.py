# from utils.binary_dataset import TokenizedBinaryDataset
# from Cdatasets.tokenizer import load_tokenizer
# tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# dataset = TokenizedBinaryDataset(
#     source="/home/prasanna/.cache/huggingface/datasets/generator/default-caaf09e7aefe9c5f/0.0.0/",           
#     tokenizer=tokenizer,
#     block_size=512,
#     bin_path="./data/automathtext.bin",  
#     num_workers=8
# )


# from utils.packed_dataset_builder import PackedDatasetBuilder

# # Load the saved binary directly into a DataLoader
# dataloader = PackedDatasetBuilder.to_dataloader(
#     bin_path="./data/automathtext.bin",
#     block_size=512,
#     batch_size=4,
#     shuffle=True,       # Shuffles blocks during training
#     num_workers=2,      # Multi-process data loading
#     drop_last=True,
#     pin_memory=True,
#     max_samples=10000
# )

# for batch in dataloader:
#     print(batch["input_ids"].shape)   # (4, 511)
#     print(batch["targets"].shape)     # (4, 511)
#     break


from Cdatasets.tokenizer import load_tokenizer
from utils.packed_dataset_builder import PackedDatasetBuilder

tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2) LOAD/READ into DataLoader
train_loader, _ = PackedDatasetBuilder.to_dataloader(
    bin_path="data/packed_automath/data.bin",
    block_size=512,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(train_loader)) 


for i in range(3): 
    # 3) SEE SAMPLE (decode first sample)
    sample_ids = batch["input_ids"][i].tolist()
    print(tokenizer.decode(sample_ids))
    print("\n\n")  # preview first 1500 chars
    print("=========================================")

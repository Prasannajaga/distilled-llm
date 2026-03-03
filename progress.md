# Purpose

I developed this model solely to understand the in-depth knowledge of how distillation works between a teacher and student model.

# Scope

What is the scope of training this from scrath?

After training an autoregressive model on the tiny-stories dataset, I always wondered "where to go next".
So I decided on a maths reasoning model trained on a part of the `math-ai/AutoMathText` dataset.
I specifically used these subsets:

1. 0.70-to-1.00
2. 0.60-to-1.00

I alaways struggled to find a good dataset when training a model. So here I choosed these two subsets which alone cost me 10GB in size.

# My Key Learnings

## Data Processing

While I always played with samll datasets with less size, I never realized how extensive tokenization on a huge corpus gets—and the complexity grows linearly.

While training my model a couple of times, I came to know that the data processing took more time than actually training the model!

After some searching, I later found a process that worked well for me to handle this huge daata:

```text
HF dataset
    ↓
Tokenize
    ↓
Pack
    ↓
Save .bin + .idx
    ↓
Memory-map
    ↓
Torch Dataset
    ↓
DataLoader
    ↓
Model
```

This reduced huge amounts of time for me honestly speaking. It took me sometime to set up, but it was completley worth it.

## Dataset Sampling

I implemented fixed dataset sampling.

There's no need to stick to just one dataset when pre-training. You can take a mixture of datasets and tokenize them together, which helps the model perform well in generallly.

## Checkpointing

I learned that saving the optimizer & scaler is necessary when you want to resume training from a checkpoint save, but you don't need them for the final checkpoint save.

But you can still store them if u don't care about size, and later split the `state_dict` alone and save it separately.

## Scaling Laws

I learned that you have to train on roughly `20 * params` tokens to get the best out of it, according to the chinchila paper's scaling laws.

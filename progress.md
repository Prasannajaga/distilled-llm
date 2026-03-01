
this model is developed soley for understand indepth knowledge of the distillation teacher & student model

what is scope of this training form scrath ?

after training the autoregressive model on tiny-stories dataset
I have always wondered where next

maths reasoning model trained part of math-ai/AutoMathText
subset used

1. 0.70-to-1.00
2. 0.60-to-1.00

I alaways struggle d to find the dataset when training model
so here I choosed this theese two subset alone cost me 10GB in size

my key learnings:

while I always played with samll datasets with less in size
I never came to know how exensize the tokenization of the huge corpus data gets and complex grows linearly

while training my model couple of times I came to know that this process tooks more than training my model

after upon some search and I
later found that process which worked for me for handle this huge daata

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

this reduced huge time for me honestly speaking.
it took me sometime but it worth it

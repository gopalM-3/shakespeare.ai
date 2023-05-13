import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1618)

# hyperparameters
batchSize = 64
blockSize = 256
maxIters = 5000
evalInter = 500
learningRate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
evalIters = 200
nEmbd = 384
nHead = 6
nLayer = 6
dropout = 0.2

# reading and viewing the text corpus
with open("./shakespeare.txt", "r") as file:
    text = file.read()

print(f"Length of the dataset: {len(text)}")
print("First 1000 characters:")
print(text[:1000])

# all the chars that occur in the doc
chars = sorted(list(set(text)))
vocabSize = len(chars)
print("Charset:", "".join(chars))
print(f"{vocabSize=}")

# creating mappings from chars to ints
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# encoder function; string to list(int)
def encode(s):
    return [stoi[c] for c in s]

# decoder function; list(int) to string


def decode(l):
    return "".join(itos[i] for i in l)


print("\n***Test***")
temp = encode("Carpe diem!")
print("Encoded vector for 'Carpe diem!':", temp)
print(decode(temp))

# encoding the entire doc
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data encoding metadata: {data.shape=}, {data.dtype=}")

# train & val split
n = int(0.9*len(data))
train = data[:n]
val = data[n:]

# loading data


def getBatch(split):
    data = train if split == "train" else val
    idx = torch.randint(len(data) - blockSize, (batchSize,))
    x = torch.stack([data[i: i + blockSize] for i in idx])
    y = torch.stack([data[i + 1: i + blockSize + 1] for i in idx])
    x, y = x.to(device), y.to(device)

    return x, y

# function for estimating losses


@torch.no_grad()
def estimateLoss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(evalIters)
        for k in range(0, evalIters):
            X, y = getBatch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

# creating necessary classes


class Head(nn.Module):
    # one head of self-attention

    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(nEmbd, headSize, bias=False)
        self.query = nn.Linear(nEmbd, headSize, bias=False)
        self.value = nn.Linear(nEmbd, headSize, bias=False)
        self.register_buffer("tril", torch.tril(
            torch.ones(blockSize, blockSize)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # computing attention weights or affinities
        wei = q @ k.transpose(-2, -1) * C**(-0.5)
        # (B, T, 16) @ (B, 16, T) --> (B, T, T)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        '''
        this is a decoder block, it prevents token communication with the future tokens
        if this decoder block is absent, the tokens will be able to communicate with the past & future tokens
        '''

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # computing weighted sum of the values
        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    # multiple heads of self-attention in parallel

    def __init__(self, numHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([Head(headSize)
                                    for _ in range(0, numHeads)])
        self.proj = nn.Linear(nEmbd, nEmbd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    # simple linear layer followed by non-linearity

    def __init__(self, nEmbd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nEmbd, 4 * nEmbd),
            nn.ReLU(),
            nn.Linear(4 * nEmbd, nEmbd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # transformer block: communication followed by computation

    def __init__(self, nEmbd, nHead):
        # nEmbd: embedding dimension, nHead: number of heads we'd like
        super().__init__()
        headSize = nEmbd // nHead
        # nHead heads of headSize dimensional self-attention
        self.sa = MultiHeadAttention(nHead, headSize)
        self.ffwd = FeedForward(nEmbd)
        self.ln1 = nn.LayerNorm(nEmbd)  # layer norm 1
        self.ln2 = nn.LayerNorm(nEmbd)  # layer norm 2

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class BigramLanguageModel(nn.Module):
    # simple bigram language model

    def __init__(self):
        super().__init__()
        # each token directly reads the logits of the next token
        self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbd)
        self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbd)
        self.blocks = nn.Sequential(
            *[Block(nEmbd, nHead=nHead) for _ in range(0, nLayer)])
        self.lnF = nn.Linear(nEmbd, nEmbd)  # final layer norm
        self.lmHead = nn.Linear(nEmbd, vocabSize)

    def forward(self, idx, targets=None):
        # idx & targets are both (B, T) tensors of integers
        B, T = idx.shape

        tokEmb = self.tokenEmbeddingTable(idx)  # (B, T, C)
        posEmb = self.positionEmbeddingTable(
            torch.arange(T, device=device))  # (T, C)
        x = tokEmb + posEmb  # (B, T, C)
        x = self.blocks(x)  # multi head attention, (B, T, C)
        x = self.lnF(x)  # (B, T, C)
        logits = self.lmHead(x)  # (B, T, C=len(vocabSize))

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # even though the model was built with the vision to use context from 1 to blockSize, this is sampling using bigrams
    def generate(self, idx, maxNewTokens):
        # idx is a (B, T) array of indices of the current context
        for _ in range(0, maxNewTokens):
            # cropping idx to the last blockSize tokens
            idxCond = idx[:, -blockSize:]
            # getting the predictions
            logits, loss = self(idxCond)
            # focusing only on the last time step (essentially making it a bigram model)
            logits = logits[:, -1, :]
            # applying softmax to get probablities
            probs = F.softmax(logits, dim=-1)
            # sampling from a multinomial distribution
            idxNext = torch.multinomial(probs, num_samples=1)
            # appending the predicted index to a running vector
            idx = torch.cat((idx, idxNext), dim=1)

        return idx


xb, yb = getBatch("train")
# the below code is to analyse & interpret the inputs and targets
# print(f"***Inputs***\nShape: {xb.shape}\n{xb}")
# print(f"***Targets***\nShape: {yb.shape}\n{yb}")

# for b in range(batchSize):
#     for t in range(blockSize):
#         context = xb[b, : t + 1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target: {target}")

model = BigramLanguageModel()
m = model.to(device)

# preliminary analysis before training
print("\n***Preliminary analysis***")
logits, loss = model(xb, yb)
print(f"{logits.shape=}, {loss=}")  # expected loss = -ln(1/65) = -4.17

print("Generated text before training:")
print(decode(model.generate(idx=torch.zeros(
    (1, 1), dtype=torch.long), maxNewTokens=100)[0].tolist()))

print("\n***Training begins***")

# creating an optimizer object
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(0, maxIters):
    # evaluating train & val loss once in a while
    if iter % evalInter == 0:
        losses = estimateLoss()
        print(
            f"{iter=}: train loss - {losses['train']:.4f}, val loss - {losses['val']:.4f}")

    # sampling a batch of data
    xb, yb = getBatch("train")

    # evaluating the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# sampling from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, maxNewTokens=100)[0].tolist()))

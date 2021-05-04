import torch
import torch.nn.functional as F
import numpy as np
import binascii
from tqdm import tqdm
import time

bits = 64
bunch_size = 16
m_bits = bits
m_bytes = bits//8
k_bits = bits
c_bits = bits
pad = 'same'

def encode_message(m):
    """
    Encodes bytes into ML inputs.
    """
    n = int(binascii.hexlify(m).ljust(m_bytes*2, b'0'), 16)
    encoded = [ (-1 if b == '0' else 1 ) for b in bin(n)[2:].rjust(m_bits, '0') ]
    assert decode_message(encoded) == m
    return encoded

def decode_message(a, threshold=0):
    """
    Decodes ML output into bytes.
    """
    i = int(''.join('0' if b < threshold else '1' for b in a), 2)
    try:
        return binascii.unhexlify(hex(i)[2:].rjust(m_bytes*2, '0'))
    except:
        return binascii.unhexlify("0"+hex(i)[2:])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.keys = torch.nn.Embedding(200,64)
        self.linear = torch.nn.Linear(m_bits+k_bits, m_bits + k_bits)
        self.linear2 = torch.nn.Linear(m_bits+k_bits, m_bits)
        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(1,2,kernel_size=4,stride=1,padding=1,padding_mode="replicate"),
            torch.nn.Sigmoid(),
            torch.nn.Conv1d(2,4,kernel_size=2,stride=2,padding=1,padding_mode="replicate"),
            torch.nn.Sigmoid(),
            torch.nn.Conv1d(4,4,kernel_size=1,stride=1),
            torch.nn.Sigmoid(),
            torch.nn.Conv1d(4,1,kernel_size=1,stride=1),
            torch.nn.Tanh(),
        )

    def forward(self, ciphertexts, keys):
        batch_size = len(ciphertexts)
        x = torch.cat([ciphertexts,keys],dim=1)
        x = self.linear(x)
        x = F.sigmoid(x)
        x = x.reshape((-1,1,m_bits+k_bits))
        x = self.convs(x)
        return x.reshape((batch_size,-1))

from pathlib import Path
plaintext = Path("philosophy.html").read_bytes()

num_chunks = 150
plaintext = torch.tensor([encode_message(plaintext[i:i+8]) for i in range(0,len(plaintext)-5,8)])[:num_chunks*16].cuda()
ciphertext = torch.tensor(np.load("ciphertext.npy")[:num_chunks*17]).cuda()

print(plaintext.shape)
print(ciphertext.shape)

encrypted_chunks = ciphertext.reshape((-1,17,64))[:,:16]
encrypted_keys = ciphertext.reshape((-1,17,64))[:,-1]
all_encrypted = encrypted_chunks.reshape((-1,64))
plaintext_chunks = plaintext.reshape((-1,16,64))

keyinds = torch.repeat_interleave(torch.arange(len(encrypted_chunks)),16).cuda()

# First train a model where we simultaneously learn
# the keys for all chunks using the known plaintext-ciphertext pairs,
# the first 103 chunks (flag is in chunk 104 and 105).
# You might need to restart this model a couple of times
# with different learning rate to get down to a loss of ~10
model = Model().cuda()
model = torch.load("model")

bestloss = 1e18
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)

for epoch in tqdm(range(0)):
    optimizer.zero_grad()
    until = 103*16
    dec = model.forward(all_encrypted[:until],model.keys(keyinds[:until]))
    loss = ((plaintext[:until] - dec)**2).sum()
    loss.backward()
    optimizer.step()
    
    if epoch%50==0:
        l = loss.detach().cpu().numpy()
        print()
        print(l)

        for j in range(100,104,1):
            print()
            dec = model.forward(encrypted_chunks[j], model.keys(torch.tensor([j]*16).cuda()))
            print(j,":",decode_message(dec.flatten().detach().cpu()))
            print(j,":",decode_message(plaintext_chunks[j].flatten().detach().cpu()))

        if l < bestloss:
            bestloss = l
            torch.save(model,"model")

        #print(model.keys(torch.tensor([1]).cpu()))


# Now we can use this model and the learned keys to decrypt the encrypted keys.
dec_keys = model.forward(encrypted_keys,model.keys(torch.arange(len(encrypted_keys)).cuda())).detach().round()

# Now we train a new model on the the known plaintext-ciphertext pairs,
# but using the decrypted keys instead of learned keys
model2 = Model().cuda()
model2 = torch.load("model2")

bestloss = 1e18
optimizer = torch.optim.AdamW(model2.parameters(), lr=0.001)

for epoch in tqdm(range(10000)):
    optimizer.zero_grad()
    until = 103*16
    dec = model2.forward(all_encrypted[16:until],dec_keys[keyinds[:until-16]])
    loss = ((plaintext[16:until] - dec)**2).sum()
    loss.backward()
    optimizer.step()
    
    if epoch%50==0:
        l = loss.detach().cpu().numpy()
        print()
        print(l)

        for j in range(100,104,1):
            print()
            #print(dec_keys[j-1].cpu().detach().numpy())
            #print(decode_message(dec_keys[j-1].cpu().detach()))
            dec = model2.forward(encrypted_chunks[j], dec_keys[torch.tensor([j-1]*16).cuda()])
            print(j,":",decode_message(dec.flatten().detach().cpu()))
            print(j,":",decode_message(plaintext_chunks[j].flatten().detach().cpu()))

        if l < bestloss:
            bestloss = l
            torch.save(model2,"model2")
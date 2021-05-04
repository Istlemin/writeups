
# Smart Crypto

From the name of the .enc file we guess that what has been encrypted is something similar to the html file at https://oooverflow.io/philosophy.html

We now have a ciphertext file which consists of encrypted 8-byte blocks where every 17th block is the encrypted key that is used for the following 17 blocks. Lets call these 17 blocks together a "chunk". Dropping the last block of each chunk gives us a set of corresponding plaintext blocks from the html file and ciphertext blocks. These are presumably correct up to some point in the html where the flag was inserted.

For each chunk we have the plaintext for, we now that there is some key and some network (with the structure of Bob) that decrypts the chunk. The network is the same for all chunks, but the key is different. However, if we simply learn all the keys and the network simultaneously, we should get some kind of network that can decrypt unseen ciphertext given a key. These keys will not be the same as the keys that are encrypted in every 17th block, but they will be equally valid for our network. Thus, we can use our learned keys and trained network to decode the encrypted keys, giving us a list of all the keys that were used during training. 

All of this is done up to chunk 103, which we found through trial and error to be the point where things start to break down.

Now the problem is that we cannot decrypt chunk 104 as we don't the key for our network for this chunk (assuming we don't know the plaintext for this chunk). However, we know what the key that worked for Bob's network, as we can decrypt this from last block of chunk 103.

Would like if we could use Bob's keys for decryption instead, so we simply train a new network on the first 103 chunks where we use Bob's keys instead of learning the keys. Now we can feed chunk 104 and the key we found in chunk 103 into this new network, giving us something like this:

```
>Fre F ags&</s rong T e`ovder of the  verflow firely  eliev7 fdagc`yearn`do be free.*Dg this efdl not gnli`are we givifg
```

From the context we can guess (with some more trial and error, guessing some part correctly might reveal more) that it should read something like

```
>Free Flags.</strong>
The order of the overflow firmly believes that flags meant to be free. To this end, not only are we giving
```

Adding this to our known plaintext and repeating everything allows us to repeat the process to decrypt chunk 105:

```
thei out over thd fseaking qhone$ but alro iere.\x02\x08ere is your fl`g:\x01OOO{turos out tiat ml csyppo is a re`mly bad idea}.*\n<p?By 
```

From which we guess the flag:

```
OOO{turns out that ml crypto is a really bad idea}
```

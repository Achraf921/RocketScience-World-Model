# Rocket Science World Model 

an attempt to re-produce at a smaller scale the Rocket League multi-agent (4) predictive World Model showcased in the 
Multiplayer Interactive World Models with Representation Autoencoders (MIRA) paper (https://arxiv.org/pdf/2607.05352)
published by General Intuition, Kyutai (French!) and Epic Games.

The demo is highkey impressive and I was not aware world models could be this accurate, I'd advice to give a look to their 
demo : https://mira-wm.com/

Currentley trying to re-implement the paper and at the moment I changed the mixing formula of the different DINOv3-L layers from the mean formular described in the paper to a trained neural net to see if this can yield anything or in contrast worsen validation loss. Goal is to get to a smaller scale implementation of the model and learn about world models 

## Current state: 

Mostly done with the Codec as a whole, the encoder's implementation is trivial since we are importing the frozen weights from Meta's DINOv3-L, but re-implementing the decoder was much harder since we had this time to provide our complete custom implementation of a ViT and handle the space and time upscaling tensor transformation which were highkey the hardest part.

### Direct next steps:

Complete the codec with a pre-processing block that splits the clips into frames and into the right tensor format our encoder consumes and does so the other way around in a non-learned way ofc so that we can actually set up the training loop and set up the
loss function and the optimizer and train


## BTW

if you are reading this and work with/have interest in world models and want to chat feel free to reach out, I don't really do linkedin that much though you can find it on my GH profile but I'll be way more responsive on Instagram at ```@92.ash0```

this is far from the final version and I'll try to update it frequently rather than doing it all on my local machine because why not
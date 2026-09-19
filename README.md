# Rocket Science World Model 

an attempt to re-produce at a smaller scale the Rocket League multi-agent (4) predictive World Model showcased in the 
Multiplayer Interactive World Models with Representation Autoencoders (MIRA) paper (https://arxiv.org/pdf/2607.05352)
published by General Intuition, Kyutai (French!) and Epic Games.

The demo is highkey impressive and I was not aware world models could be this accurate, I'd advice to give a look to their 
demo : https://mira-wm.com/

Currentley trying to re-implement the paper and at the moment I changed the mixing formula of the different DINOv3-L layers from the mean formular described in the paper to a trained neural net to see if this can yield anything or in contrast worsen validation loss. Goal is to get to a smaller scale implementation of the model and learn about world models 

## Current state: 

Completley done with the codec, data loader and training loop end to end, only issue is that we are not able to debug on my machine since it won't run (cpu nor mps) we uped the dataset to 6 shards (18GB), we also tied down the passing arguments from each sub-block of the codec in a cascading/graph way such that we now exert full control over the size of the model through its hyperparameters.
Ran a 10 round training test (MPS is acting overly weird and slow compared to what I experienced before) and got outputs that I could not distinguish from noise, which is not too unexpected
<img src="./training_output//20260919_123750/samples/step_000009.png" width = "400px">

### Direct next steps:

rent compute


## BTW

if you are reading this and work with/have interest in world models and want to chat feel free to reach out, I don't really do linkedin that much though you can find it on my GH profile but I'll be way more responsive on Instagram at ```@92.ash0```

this is far from the final version and I'll try to update it frequently rather than doing it all on my local machine because why not
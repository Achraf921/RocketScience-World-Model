import torch
from codec import Codec, RLClips
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os, time

# where the training happens

#!! We assume the output data dirs were already create (manually) if not, this will create them

run = f"training_output/{time.strftime('%Y%m%d_%H%M%S')}" # timestamped runs btw
os.makedirs(f"{run}/samples", exist_ok=True)
os.makedirs(f"{run}/checkpoints", exist_ok=True)
log = open(f"{run}/log.txt", "a")

# btw this was not planned for parallel multi-GPU training, might be a cool add on to do later if I ever get more time

#------ some hyper params (most are specified within the model such as lr, schedule, format, go check codec.py)

batch_size = 1 # 32 per the paper
total_steps = 100 # 249 000
warmup_steps = 20 #1000
min_lr = 1e-6
max_lr = 2e-4
num_workers = 8 
checkpoint = 10000 # every checkpoint, steps, we save a snapshot of the weights
device = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


#------ some hyperparameters to scale down the model for M4, 24GB test training, discard/comment_out for the real run
decoder_mlp_mult = 4
T = 40
n_head = 12
depth = 12
n_embd = 768

# -------- model
codec = Codec(batch_size=batch_size, total_steps=total_steps, min_lr=min_lr, max_lr=max_lr, decoder_mlp_mult=decoder_mlp_mult, n_head=n_head, depth=depth, T=T, n_embd=n_embd, warmup_steps=warmup_steps).to(device)

# just to get parameter count before a run
paramlist = []
    
nparams = nparams = sum(p.numel() for p in codec.parameter_list)

print(f'Parameter count : {nparams}')
print(f'Device : {device}')

#import sys; sys.exit(0)

# we start by loading the training data
ds = RLClips("data/rocket/train/unpacked", T=T)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
it = iter(dl)
# iterator of tensors [32, 40, 3, 720, 1280], 30672 of them exactly from our current data

ds_test = RLClips("data/rocket/test/unpacked", T=T)
dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
it_test = iter(dl_test)

# we can now properly iterate :

for step in range(total_steps):
    try:
        batch = next(it).to(device)
    except StopIteration: # epoch ended
        it = iter(dl)
        batch = next(it).to(device)
    #forward pass on the batch

    codec.train() # switching bool on
    x = batch.flatten(0, 1) # (T*B, C, H, W)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = codec.pre_processor(x)
        enc, intermediate = codec.encoder(x, is_training=True) # such that we get both output and the layers we need
        y = codec.decoder(enc)
        # bakward pass on loss
        loss = codec.loss(x,y,intermediate_layers=intermediate)
    lr = codec._get_cosine_lr(step)
    codec.optimizer.zero_grad()
    loss.backward()
    for g in codec.optimizer.param_groups: 
        g['lr'] = lr
    codec.optimizer.step()
    # Training loss
    print(f'Step : {step}, Training Loss : {loss:.4f}')
    log.write(f"train,{step},{lr:.2e},{loss.item():.4f}\n"); log.flush()

    # every 250 steps, compute validation loss: 
    if step%250 == 0 or step == total_steps-1:
        try:
            test_batch = next(it_test).to(device)
        except StopIteration: # epoch ended
            it_test = iter(dl_test)
            test_batch = next(it_test).to(device)
        codec.eval() # switching bool off
        x = test_batch.flatten(0, 1) # (T*B, C, H, W)
        x = codec.pre_processor(x)
        enc, intermediate = codec.encoder(x, is_training=True) # such that we get both output and the layers we need
        y = codec.decoder(enc)
        # compute loss
        val_loss = codec.loss(x,y,intermediate_layers=intermediate)
        print(f'Validation Loss : {val_loss.item():.4f}')
        log.write(f"val,{step},{lr:.2e},{val_loss.item():.4f}\n"); log.flush()

    # every 1000 steps, save the batch's first 10 images before and after
    if step%500 == 0 or step == total_steps -1:
        x_first10 = codec.post_processor(x[:10,:,:,:]).float() * (1/255)
        y_first10 = codec.post_processor(y[:10,:,:,:]).float() * (1/255) # we finally use post processor lol
        pair = torch.cat([x_first10, y_first10], dim=3)
        save_image(pair, f"{run}/samples/step_{step:06d}.png", nrow=1)

    if step % checkpoint == 0 or step == total_steps-1: #saving the weights on the last iteration and every checkpoint
        torch.save({
            "step": step,
            "model": {k: v for k, v in codec.state_dict().items() if not k.startswith(("encoder.dino", "lpips"))},
            "optimizer": codec.optimizer.state_dict(),
        }, f"{run}/checkpoints/step_{step:07d}.pt")


'''
Hyperparameters I used for the local test :

batch_size = 1 
total_steps = 10 
warmup_steps = 1
min_lr = 1e-6
max_lr = 2e-4
num_workers = 0
checkpoint = 100000
decoder_mlp_mult = 1
T = 16
n_head = 4
depth = 4
n_embd = 256

Hyperparameters I used for the real training run :


batch_size = 1
total_steps = 32000
warmup_steps = 1000 
min_lr = 1e-6
max_lr = 2e-4
num_workers = 8 
checkpoint = 2000
decoder_mlp_mult = 4
T = 40
n_head = 12
depth = 12
n_embd = 768

with 50 train shards and 4 val shards downloaded


'''

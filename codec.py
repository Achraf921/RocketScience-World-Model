import torch
import torch.nn as nn
from transformers import AutoModel
import math
import torch.nn.functional as F
import lpips
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import VideoDecoder
import glob, os
import lpips


# ---- Codec ------- : (btw Codec does contain the training loop which contains)

#!! BTW: we tried out best to do two things right regarding the software design here :
#   1) We ensured that the hardcoded values are the ones matching the paper's Table 9 : (https://arxiv.org/pdf/2607.05352)
#   2) We ensured that if you chose in the driver to explicitely pass down a value, it will go down to where it is supposed 
#      to go in a coherent argument passing graph between all of our sub-modules

class Codec(nn.Module):
    def __init__(self, max_lr=2e-4, b1b2=(0.9,0.95), wdecay=0, warmup_steps=1000, min_lr=1e-6, total_steps=249000, batch_size=32, n_embd=1152, T=40, n_head=16, depth=28, decoder_mlp_mult=4):
        # out sub-blocks
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(n_embd=n_embd, T=T//2, n_head=n_head, depth=depth, vit_mlp_mult=decoder_mlp_mult) # we divide by two
        self.pre_processor = pre_processor()
        self.post_processor = post_processor()

        #parameter list
        self.parameter_list = list(self.decoder.parameters()) + [p for p in self.encoder.parameters() if p.requires_grad]
        # since some elements of the encoder don't require grad such as the dinoV3-L

        # optimizer
        self.optimizer = torch.optim.AdamW(params=self.parameter_list, betas=b1b2, weight_decay=wdecay) # eps is default, paper specifies AdamW but not the lr schedule which we will assume cosine like in GPT-2
        self.warmup_steps=warmup_steps
        self.total_steps=total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lpips = lpips.LPIPS(net="vgg").eval().requires_grad_(False) # as mentioned in the paper (blackbox implementation lowkey, I did NOT read that paper yet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
        # hardcoded means and std from ImageNet
        self.lambda_msd = self.lambda_lpips = torch.tensor(1.0)

    # we need to implement a couple inner functions

    def _get_cosine_lr(self, step): # at step : step
            if step<=self.warmup_steps:
                # linear warmup function
                return step * ((self.max_lr - self.min_lr)/self.warmup_steps) + self.min_lr
                # f(0) = min_lr
                # f(100) = max_lr
                # f(x) = ax+b
                # -> b = min_lr
                # warmup*a + min_lr = max_lr <=> a = (max_lr - min_lr)/warmup
                # f(step) = (step * (maxlr-minlr)/warmup) +minlr # linear warmup function
            elif step>self.total_steps: # should not happen but we still put it
                return self.min_lr
            else:
                decay_ratio = (step-self.warmup_steps) / (self.total_steps-self.warmup_steps) # as computed in the GPT-2 rebuild
                assert 0<=decay_ratio<=1
                coeff = 0.5*(1.0+math.cos(math.pi*decay_ratio))
                return self.min_lr + coeff * (self.max_lr-self.min_lr) # all from the GPT-2 buildout btw

    def _imagenet_unnormalization(self, x):
        # x of shape [T, C, H, W] with C = 3 (R,G,B)
        norm = x*self.std.to(x.device)
        norm = norm + self.mean.to(x.device)
        norm = norm*2
        norm = norm -1 # to shift from [0,1] to [-1, 1] as expected by lpips
        return norm
                
    def loss(self, x, y, intermediate_layers):
        # per the paper, loss seems to have 3 components, a regular L1 component, a LPIPS component, and a re-encode through the DINOV3-L
        # loss component that re-encodes the output and computes a L1 on different intermediate layers
        # loss is averaged over all components in all dimensions and the returned as a single valye on which we'll call backwards
        # x input is of shape (B*T, C, H, W) # B being the batch size, we put all the frames on the T scale when batched
        # y is of the shape (B*T, C, H, W) and si the output of the decoder without post_processing applied

        #! BTW, the two non-l1 components only ever get computed on 25% of the frames in a batch,

        N = x.shape[0] # number of frames passed in the batch
        mask = torch.rand(N, device=x.device) < 0.25 # we'll use this to mask x and y and only get 25% of the frames for the non-l1 component loss element

        # let's first compute the l1 loss component

        l1_component = F.l1_loss(x,y)

        # now we need the intermediate layer's output, to get that we will implement three changed to the decoder :
        # 1) goal will be to cache the intermediate layers that we used for inference (the exact layers are not specified but this 
        # decision makes sense for the sake of efficiency)
        # 2) we'll add a is_training boolean flag to the encoder (default False) such that it only stores those values for training and not for inference
        # 3) obv make sure to pass them into loss so that we don't have to re-compute them
        # btw they're of format : [F, T, H//16, W//16, n_embd] or here , [7, T, 18, 32, 1024]
        # i.e before the bottlneck that halves T, H, W
        # btw it is not l1 loss that is applied to them but mean of squared differences
        # and we also need to re-encode the output's y into the encoder's feature extractor and grab the intermediate dimensions 
        # in the same way we've done at inference time
        if mask.any():
            il_output = self.encoder.dino(pixel_values=y[mask], output_hidden_states=True) # taking in the mask
            il_output = [il_output.hidden_states[i+1] for i in self.encoder.layers] # it is good that in python, class member access if straighforward
            il_output = torch.stack([f[:,5:,:] for f in il_output]) # code from the forward of encoder btw
            il_output = il_output.view(intermediate_layers[:,mask].shape) # to be sure we're in the same shape

            intermediate_msd = F.mse_loss(il_output, intermediate_layers[:,mask]) # taking in the mask
         
        # now we need to implement the lpips component

            lpips_component = self.lpips(self._imagenet_unnormalization(x[mask]),self._imagenet_unnormalization(y[mask])).mean() # also masked on 25% of frames in the batch

        # gains
            eps = 1e-4
            w = self.decoder.token_scale_conv.weight
            if torch.is_grad_enabled():
                g_rec = torch.autograd.grad(l1_component, w, retain_graph=True)[0].norm()
                g_msd = torch.autograd.grad(intermediate_msd, w, retain_graph=True)[0].norm()
                g_lp  = torch.autograd.grad(lpips_component, w, retain_graph=True)[0].norm()
                lambda_msd   = (g_rec / (g_msd + eps)).detach()
                lambda_lpips = (g_rec / (g_lp + eps)).detach()
                self.lambda_msd, self.lambda_lpips = lambda_msd, lambda_lpips    

            else:
                lambda_msd, lambda_lpips = self.lambda_msd, self.lambda_lpips

            loss = l1_component + lambda_msd*intermediate_msd + lambda_lpips*lpips_component # final loss 

        else: #scenario of an mask that yeilds an empty tensor (errors)
            loss = l1_component

        return loss

    def forward(self, x): # x of shape [T,C,H,W]
        out = self.pre_processor(x)
        out = self.encoder(out)
        out = self.decoder(out)
        return out # lowkey irelevant for now, maybe I should change it when the world model will be plugged into encode and decode functions
        
    def train(self, mode=True):
        super().train(mode)
        self.lpips.eval()
        self.encoder.dino.eval()
        return self

# ---- Encoder ------

class Encoder(nn.Module):

    def __init__(self, layers=(11, 13, 15, 17, 19, 21, 23), C=32, H=288, W=512, n_embd=1024): # ugly number of layers
        super().__init__()
        # btw 1024 is dino's encoding dim, not ours, ours is 1152
        self.dino = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', attn_implementation="sdpa") # import DINO model, attn impl swaps manual atn for F.scaled_dot_product_attention
        self.dino.eval().requires_grad_(False) # making sure we don't backprop/optimize through dino (300M params)
        # btw for ViT-L, all embedding spaces have 1024 channels, hence : 
        self.layers = layers
        self.n_embd = n_embd
        self.C = C
        self.H = H
        self.W = W
        k = len(layers)
        self.mix = nn.ParameterList([nn.Parameter(torch.full((H//16, W//16, n_embd),1/k)) for _ in range(k)]) # mixing into a single embedding space
        self.bottleneck = nn.Conv3d(self.n_embd, self.C, kernel_size=2, stride=2)
         # times 8 since we basically divide by 1/2 * 1/2 * 1/2 the other dims, such that 8/8 = 1

    def _bottleneck(self, t):
        T, H, W, emb = t.shape
        output = t # we don't mind sharing pointers since the input is just inner state before we reach the latent space
        output = output.permute(3, 0, 1, 2) # here permutating dim 0 and 3 does the rotation I tried to describe which leaves us with 1024 18*32 tensors x Time
        output = self.bottleneck(output) # applying trainable convolutional network to merge the 3d - chunks into a single value
        output = output.permute (1,2,3,0) # permuting everything back into order
        return output

    def forward(self, frames, is_training=False):
        # assuming we get a frames tensor of shape [T, C, W, H] as input already pre-formated with H and W being respectively 288 and 512
        T, C, H ,W = frames.shape
        with torch.no_grad():
            features = self.dino(pixel_values=frames, output_hidden_states=True) # getting all hidden state
            # features.hidden_state is a tuple of [T, 581, 1024]
            selected_features = [features.hidden_states[i+1] for i in self.layers] # all the layers we want for layer mixing
            # an array of (T, 581, 1024) tensor with 5 extra tokens we need to clean off
            # that all sit at the beginning so easy to clean off
            selected_features = torch.stack([f[:,5:,:] for f in selected_features]) # stack to concat the array into a single tensor
            selected_features = selected_features.view(len(self.layers), T, self.H//16, self.W//16, self.n_embd) #viewing it accordingly
        # now we can mix em up
        mixed = torch.zeros_like(selected_features[0])
        for f,w in zip(selected_features, self.mix):
            mixed+= f * w #stacking up the dim=0 dim through additiong
        # we end up with a mixed in shape (T, H//16, W//16, self.n_embd)
        # we can now simply bottleneck it as we defined to get it to the latent space
        latent_space = self._bottleneck(mixed) # finally
        if is_training is True:
            return latent_space, selected_features # for training, in [F, T, H//16, W//16, n_embd]
        else:
            return latent_space


# ------ Decoder ---------

class Decoder(nn.Module):
   def __init__(self, T=20, H = 9, W= 16, C=32, n_embd=1152, depth=28, vit_mlp_mult=4, n_head=16):
      super().__init__()
      self.T = T
      self.H = H
      self.W = W
      self.C = C
      self.n_embd = n_embd
      self.spatial_upsample = nn.Linear(C, n_embd*4)
      # ourput will be in the following format : (T, 2H, 2W, n_embd) (i.e 10, 18, 32 , 1152)
      self.vit = ViT(T=T, n_embd=n_embd, depth=depth, mlp_mult=vit_mlp_mult, n_head=n_head) # default hardcoded values are correct for the paper's hyperparams for now
      self.time_upsample = nn.Linear(n_embd, n_embd*2) #doubling, shared
      self.token_scale_conv = nn.ConvTranspose2d(n_embd, 3, kernel_size=16, stride=16) # as mentioned in the paper, each token in the 228*512 grid embedds a 16x16 pixel patch in the 
      # real image, and each pixel embedds RGB, i.e 3 uint8 values, hence why the 16*16*3 final embedding, we use a convolutional network to unsplit the 768 embd into 16x16 tokens


   def _spatial_upsample(self, t):
      out = t
      out = self.spatial_upsample(out) # growing back the last dimension
      # doing the view operation we mentioned above to have the H x W matrix go from storing 4608 vectors to 2 x 2 sub matrices 
      # of 1024 elements each in its cells
      out = out.view(self.T, self.H, self.W, 2, 2, self.n_embd)
      out = torch.permute(out, (0,1,3,2,4,5)) # doing the permutations to get a ((H, 2), (W, 2)) in the middle
      out = out.reshape(self.T, self.H, 2, self.W*2, self.n_embd) # one cool thing about torch.reshape is that we don't need to isolate the 
      out = out.reshape(self.T, self.H*2, self.W*2, self.n_embd)

      return out
   
   def _time_upsample(self, x):
      T, H, W, n_embd = x.shape
      out = x.reshape(T,-1,W,n_embd) # stacking all frames along the row dimension, we hence stack all frames on height which yeilds (18x20, 32, 1152), good
      out = self.time_upsample(out) # doubling all embeddings, through shared matmul
      out = out.view(T,H,W,2,n_embd) # spliting the new embeddings into two (20, 18 32, [1152, 1152]) without touching memory order
      # where we now basically have the 2 dimensionality indexing between the parent frames and the children frames (21 and up)
      out = out.permute (0, 3, 1, 2, 4) # we need to permute to be able to properly use reshape on the height dim (i.e bring the 
      #[1152,1152] tensors to the right side of the time frame of the matrix so that we get (20, 2, 18, 2, 32, 1152)
      out = out.reshape(2*T, H, W, n_embd) # we want to time dimension to eat those tuples in 2
      return out # those are always so tricky to execute, idk how many times I re-wrote this block to get to the semantically correct one

   def forward(self, x):
      #x should be of shape (20, 9, 16, 32), we will first upscale it as seen in the _spatial_upsample
      out = self._spatial_upsample(x)
      # out should now be of (20, 18, 32, 1152) shape which we will put through our built ViT
      out = self.vit(out)
      out = self._time_upsample(out)
      out = self.token_scale_conv(out.permute(0,3,1,2)) # moving down to 768 and then reverse conv our way to 16x16 chunks bilt on those 768 embeddigns
      # btw we need to permute here as convTranspose2d expects the channel to be first [T, 3, H,W]
      
      return out #finally, at [T, 3, H, W] format, we should be more than good


# -------- Encoder dependencies (ViT, Attention mechanisms, un-patchification mechanisms, ...)

class ViT(nn.Module): # the paper specifies in table 9 a depth of 28 so 28 (attention->mlp) blocks stacked, let's just specify the blocks
   def __init__(self, depth=28, T=20, H=18, W=32, n_embd=1152, mlp_mult=4, n_head=16):
      super().__init__()
      self.blocks = nn.ModuleList([Block(T=T, n_embd=n_embd, mlp_mult=mlp_mult, n_head=n_head) for _ in range(depth)]) # instantiating depth-long array of blocks
      self.positional_embeddings = nn.Parameter(torch.randn((T,H,W,n_embd))*0.02) # learned positional embeddings, applied once  before as specified in the transformer architecture
      #scaling down the positional embeddings as seen in GPT-2 

   def forward(self, x):
      # once again, with (T, H, W, n_embd) shaped input, we'll sequentially apply the blocks
      out = x + self.positional_embeddings
      for block in self.blocks:
         out = block(out)
      return out


class Block(nn.Module):
   def __init__(self, n_embd=1152, T=20, mlp_mult=4, n_head=16):
      super().__init__()
      self.attention = Attention(n_embd=n_embd, T=T, n_head=n_head) # we hardcoded the paper's default values correctley so no need to specify anything for now unless we would later like to change a hyperparameter
      self.mlp = MLP(n_embd=n_embd, mlp_mult=mlp_mult)

   def forward(self,x): # here again, input should be of format (T,H,W,n_embd)
      out = self.attention(x)
      out = self.mlp(out) 
      # btw all pre-norm layer normalization and residual connections are implemented within the attention and mlp blocks so no need to do anything here
      return out



class MLP(nn.Module): 
   def __init__(self, n_embd=1152, mlp_mult=4):
         super().__init__()
         # input comes as (T,H,W,n_embd) and we work with a 4x multiplier on n_embd
         self.n_embd = n_embd
         self.mlp_mult = mlp_mult
         self.ln = nn.LayerNorm(n_embd) # pre-norm ln learned gammas and betas
         self.layer1 = nn.Linear(n_embd, n_embd*mlp_mult) # scaling up to the MLP's dim
         self.non_linearity = nn.GELU() # non-linearity
         self.layer2 = nn.Linear(n_embd*mlp_mult, n_embd) # scaling back down to the embd's dim

   def forward(self, x):
      out = self.ln(x) # pre-norm
      out = self.layer1(out)
      out = self.non_linearity(out) #GELU
      out = self.layer2(out)
      out = out + x # residual connection
      return out

# works with (T, H, W n_embd) inputs and returns (T, H, W, n_embd) with space and time attention applied sequentially
class Attention(nn.Module):
   def __init__(self, n_head=16, n_embd=1152, T=20):
      super().__init__()
      self.n_embd=n_embd
      self.head_dim = n_embd//n_head
      self.space_heads = nn.ModuleList([SpaceAttentionHead(n_head=n_head, T=T, n_embd=n_embd) for _ in range(n_head)])
      self.time_heads = nn.ModuleList([TimeAttentionHead(n_head=n_head, T=T, n_embd=n_embd) for _ in range(n_head)])
      self.space_ln = nn.LayerNorm(n_embd)
      self.time_ln = nn.LayerNorm(n_embd)  # since we'll work with the output of the 1st space attention layer
      self.space_mixing = nn.Linear(n_embd, n_embd)
      self.time_mixing = nn.Linear(n_embd, n_embd) # linear layers we'll use to mix the concatenations 

   def forward(self, x):
      # here x is of dim (T, H, W, n_embd)
      # we need first to apply spatial attention to all frames and then time attention too
      T, H, W, emb = x.shape
      out = self.space_ln(x) # pre-norm layer norm
      if torch.is_autocast_enabled():
          out = out.to(torch.get_autocast_dtype("cuda"))
      out = torch.concat([h(out) for h in self.space_heads], dim=-1).reshape(T,H,W,self.n_embd) # applying space attention, we also let a residual connection in
      # reshaping into 2d output for us to have coherent space and time attention head code and to be consistent
      out = self.space_mixing(out) #mixing
      out = out + x #residual connection for space attention (2d)
      newx = out # output of the space attention is the new input for the time attention
      out = self.time_ln(newx) # 2nd pre-norm layernorm
      if torch.is_autocast_enabled():
          out = out.to(torch.get_autocast_dtype("cuda"))
      out = torch.concat([h(out) for h in self.time_heads], dim=-1).reshape(T,H,W,self.n_embd) # stacking the columns
      out = self.time_mixing(out)
      out = out + newx # residual connection for time attention
      return out # simple this time


class SpaceAttentionHead(nn.Module): # not a join space-time attention, we are applying two reqular attention QK sequentially rather than a 3 dimensional one
   def __init__(self, H=18, W=32 , n_embd=1152, n_head=16, T=20):
      super().__init__()
      self.T = T
      self.H = H
      self.W = W
      self.n_embd = n_embd
      head_dim = n_embd//n_head
      self.wQ = nn.Linear(n_embd, head_dim)
      self.wK = nn.Linear(n_embd, head_dim) # @ matmul dot product to get the QK (H*W, H*W) shape
      self.wV = nn.Linear(n_embd, head_dim) # values to be multiplied to the layernormed, softmaxed attention through matmul again
      # all learned
      # finally the layer normalization learned gamma and betas 
      self.q_ln = nn.LayerNorm(head_dim)
      self.k_ln = nn.LayerNorm(head_dim)
   def forward(self, x):
      # x is of shape (H,W,n_embd)

      x = x.reshape(self.T, self.H*self.W, self.n_embd) # flatening the tokens (576 tokens)
      Q = self.wQ(x)
      K = self.wK(x)
      V = self.wV(x)
      # we now apply our layer norms real quick 
      Q = self.q_ln(Q)
      K = self.k_ln(K)
      out = F.scaled_dot_product_attention(Q, K, V)
      return out

class TimeAttentionHead(nn.Module):
   def __init__(self, n_head=16, n_embd=1152, T=20, H=18, W=32):
      super().__init__()
       # so here, again, we'll be working with (T, H, W, n_embd) tensor
      self.emb_head = n_embd//n_head
      self.wQ = nn.Linear(n_embd, self.emb_head)
      self.wK = nn.Linear(n_embd, self.emb_head)
      self.wV = nn.Linear(n_embd, self.emb_head)
      # we also need our layer norm layers for the QK normalization applies in layer norm
      self.ln_q= nn.LayerNorm(self.emb_head)
      self.ln_k= nn.LayerNorm(self.emb_head)

   def forward(self, x):

      T,H,W,n_embd = x.shape

      x = x.reshape(T, H*W, n_embd) # automatically picks the right column to H which is W so it works here natively
      # now we want to permute
      x = x.permute(1,0,2) # going to (H*W, T, n_embd) (i.e, [frame1, frame2, frame3,....]) each frame being made of N*W tokens of embedding n_embd
      # and now we may want to generate our 
      Q = self.wQ(x)
      K= self.wK(x)
      V = self.wV(x)
      # Q.shape, K.shape, V.shape = (H*W, T, head_emb)
      # QK norm
      Q = self.ln_q(Q)
      K = self.ln_k(K)
      # now we need to mask
      out = F.scaled_dot_product_attention(Q, K, V, is_causal=True) 
      out = out.permute(1,0,2) # going back to frames first
      return out

# -------- Codec dependencies (pre and post processors, dataloaders, ...) ---------

class pre_processor(nn.Module): # parameter less, takes [T, 3, 1280, 720] down dimension wise on H and W
    def __init__(self):
        super().__init__()
        #p = AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
        #mean = p.image_mean is (0.485, 0.456, 0.406)
        #std = p.image_std is (0.229, 0.224, 0.225)
        # here it is ok to hardcode them since the weights are frozen
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1) # we reshape for the broadcasting to happen as expected
        # since broadcasting aligns from the right, we want 3 to land on the 3 in [T,3,H,W]


    def forward(self, x): # single function call on [T, 3, 1280, 720]-shaped x
        out = torch.nn.functional.interpolate(x.float(), size=(288, 512), mode="bilinear", antialias=True)# interpolate expects channels at 1 so we are already good on that
        out = out * (1/255) # normalizing to get 0,1 values
        out = (out - self.mean.to(x.device))/self.std.to(x.device) # according to broadcasting rules, this should work just fine
        return out # we return a [T, 3, 288, 512] shaped tensor

class post_processor(nn.Module): # also parameter less
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
        # same as above

    def forward(self, x): # x is the decoder's output of shape [T, 3, 288, 512]
        # let's first reverse mean-std normalization
        out = x*self.std.to(x.device)
        out = out + self.mean.to(x.device)
        out = out.clamp(0,1)
        out = torch.nn.functional.interpolate(out, size=(720, 1280), mode="bilinear") # same as above, just to scale up this time
        out = out * 255
        out = out.round() 
        out = out.to(torch.uint8) # rgb's dtype
        # we are now in [T, 3, 720, 1280]
        return out

class RLClips(Dataset):
    def __init__(self, root, T=40, views=(0, 1, 2, 3)):
        self.T = T
        self.entries = []
        for meta in sorted(glob.glob(os.path.join(root, "*.meta.json"))):
            key = meta[:-len(".meta.json")]
            for p in views:
                for start in (0, T): # two windows per 80-frame chunk
                    self.entries.append((key, p, start))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        key, p, start = self.entries[i]
        dec = VideoDecoder(f"{key}.p{p}.mp4")
        frames = dec[start : start + self.T]    # (T, 3, H, W) uint8, already TCHW
        return frames

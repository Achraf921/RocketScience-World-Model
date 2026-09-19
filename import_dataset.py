# to only run once if you don't already have the shards on disk that we will need for training

from huggingface_hub import snapshot_download
import os, tarfile

N_TRAIN, N_TEST = 50, 6
root = './data/rocket'

patterns = ['train/index.json', 'test/index.json'] + [f'train/dataset_{i:05d}.tar' for i in range(N_TRAIN)] + [f'test/dataset_{i:05d}.tar'  for i in range(N_TEST)]

snapshot_download(
    "kyutai/rocket-science",
    repo_type='dataset',
    allow_patterns=patterns,
    local_dir = './data/rocket'
)
# takes around 30-35 mins, you can give it a scroll

train_files = [f'{root}/train/dataset_{i:05d}.tar' for i in range(N_TRAIN)]
dst_train = './data/rocket/train/unpacked'

test_files = [f'{root}/test/dataset_{i:05d}.tar' for i in range(N_TEST)]
dst_test = './data/rocket/test/unpacked'

os.makedirs(dst_train, exist_ok=True)
os.makedirs(dst_test, exist_ok=True)

for file in train_files:
    with tarfile.open(file) as ball:
        ball.extractall(dst_train,filter='data')

for file in test_files:
    with tarfile.open(file) as ball:
            ball.extractall(dst_test,filter='data')


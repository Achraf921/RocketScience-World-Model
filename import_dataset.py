# to only run once if you don't already have the shards on disk that we will need for training

from huggingface_hub import snapshot_download
import os, tarfile

snapshot_download(
    "kyutai/rocket-science",
    repo_type='dataset',
    allow_patterns = ['train/index.json', 'train/dataset_00000.tar', 'train/dataset_00001.tar', 'train/dataset_00002.tar','train/dataset_00003.tar', 'train/dataset_00004.tar','train/dataset_00005.tar',
                        'test/index.json', 'test/dataset_00000.tar'], # split : 6 train shards, 1 test shard
    local_dir = './data/rocket'
)
# takes around 30-35 mins, you can give it a scroll

file1 = './data/rocket/train/dataset_00000.tar'
file2 = './data/rocket/train/dataset_00001.tar'
file3 = './data/rocket/train/dataset_00002.tar'
file4 = './data/rocket/train/dataset_00003.tar'
file5 = './data/rocket/train/dataset_00004.tar'
file6 = './data/rocket/train/dataset_00005.tar'
file7 = './data/rocket/test/dataset_00000.tar'
train_files = [file1, file2, file3, file4, file5, file6] # feel free to add as many shards as you need/can
dst_train = './data/rocket/train/unpacked'

test_files = [file7]
dst_test = './data/rocket/test/unpacked'

os.makedirs(dst_train, exist_ok=True)
os.makedirs(dst_test, exist_ok=True)

for file in train_files:
    with tarfile.open(file) as ball:
        ball.extractall(dst_train,filter='data')

for file in test_files:
    with tarfile.open(file) as ball:
            ball.extractall(dst_test,filter='data')


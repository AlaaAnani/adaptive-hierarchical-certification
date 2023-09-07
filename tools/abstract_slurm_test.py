import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data.distributed import DistributedSampler

# print(torch.cuda.get_device_properties(0))




class ds(Dataset):
    def __init__(self, N=1000):
        self.N = N
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return np.zeros((1, 1024, 2048, 19))


class Inf:
    def __init__(self, rank, nprocs, jobid) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '17787'
        print(f'Initializing {rank}/{nprocs}, jobid={jobid}')
        d = ds()
        self.rank = rank

        datasampler = DistributedSampler(
            dataset=d,
            rank=rank,
            num_replicas=nprocs)
        
        self.dl = DataLoader(
            dataset=d, 
            batch_size=1,
            num_workers=1,
            sampler=datasampler,
            drop_last=False
            )
        self.run()

    def run(self):
        print('Testing 1000 samples')
        samples = np.zeros((8000, 1024*2048, 19), dtype=np.float16)
        print('after creating the samples', samples.shape)
        # for x in tqdm(self.dl):
        #     # x = x.to(f'cuda:{self.rank}')
        #     continue
        print(f'Done {self.rank}')
        



if __name__ == "__main__":	 
    parser = ArgumentParser()
    parser.add_argument("--g", default='100')
    args, _ = parser.parse_known_args()
    print('args.g', args.g)

    print(f'echo ${{CUDA_VISIBLE_DEVICES}}')
    print('os.system', os.popen(f'echo ${{CUDA_VISIBLE_DEVICES}}').read())

    print('Device count', torch.cuda.device_count()) 	
    remote = False
    if not remote:
        Inf(0, 1, 0)
    else:
        torch.multiprocessing.spawn(
            Inf,
            nprocs=torch.cuda.device_count(),
            args=(torch.cuda.device_count(), args.g),
            join=True)
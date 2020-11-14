import torch
import tqdm
import musdb
import numpy as np

class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, target, samples_per_track, seq_duration, dtype):
        self.mus = musdb.DB(root = root, split = split)
        self.split = split
        self.samples_per_track = samples_per_track
        self.seq_duration = seq_duration
        self.dtype = dtype
        self.target = target
        
    def __getitem__(self, index):
        audio_sources = []
        track = self.mus.tracks[index // self.samples_per_track]
        # at training time we assemble a custom mix
        if self.split == 'train' and self.seq_duration:
            for k, source in enumerate(self.mus.setup['sources']):
                # memorize index of target source
                if source == self.target:
                    target_ind = k
                
                # set the excerpt duration
                track.chunk_duration = self.seq_duration
                # set random start position
                track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
                # load source audio and apply time domain source_augmentations
                audio = torch.tensor(track.sources[source].audio.T, dtype = self.dtype)
                audio_sources.append(audio)
            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup['sources'].keys()).index('vocals')
                # apply time domain subtraction
                y = x - stems[vocind]
        
        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.tensor(
                track.audio.T,
                dtype = self.dtype
            )
            y = torch.tensor(
                track.targets[self.target].audio.T,
                dtype = self.dtype
            )
        
        return x, y
    
    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track

    
def load_dataset(root, samples_per_track, target, seq_duration, dtype):
    train_dataset = MUSDBDataset(
        root = root,
        split='train',
        samples_per_track = samples_per_track,
        seq_duration = seq_duration,
        target = target,
        dtype = dtype
    )

    valid_dataset = MUSDBDataset(
        root = root,
        split='valid',
        samples_per_track=1,
        seq_duration = None,
        target = target,
        dtype = dtype
    )

    return train_dataset, valid_dataset
    
   
   
if __init__ == "__main__":
    train_dataset, test_dataset = load_dataset(root = "~/MUSDB18/MUSDB18-7",
                                           samples_per_track = 2,
                                           seq_duration = 5,
                                           dtype = torch.float32,
                                           target = "vocals")
    total_training_duration = 0
    for k in tqdm.tqdm(range(len(train_dataset))):
        x, y = train_dataset[k]
        total_training_duration += x.shape[1] / 44100 #train_dataset.sample_rate
    
    print ("Total Training Duration : {}".format(total_training_duration))
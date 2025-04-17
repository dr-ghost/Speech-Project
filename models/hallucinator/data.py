import os
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torch.nn as nn

import subprocess
import torchaudio
import torchaudio.transforms as T
import gdown
import random
import numpy as np

from tqdm import tqdm

from pathlib import Path


DATASET_DIR = Path(__file__).parent.absolute()/'dataset/'

class VCTKDataset(Dataset):
    """
    PyTorch Dataset for the VCTK corpus.
    
    Assumes the VCTK directory structure is like:
    VCTK-Corpus-0.92/
      ├── wav48/
      │    ├── p225/
      │    │     ├── p225_001.wav
      │    │     ├── p225_002.wav
      │    │     └── ...
      │    ├── p226/
      │    │     ├── p226_001.wav
      │    │     └── ...
      │    └── ...
      └── txt/    (optional transcripts if available)
    """
    def __init__(self, root_dir: str, subset: str="wav48", transform=None, wavlm_fn=None, wavlm: nn.Module=None, device: torch.DeviceObjType=None) -> None:
        """
        Args:
            root_dir (str): Path to the root directory of the VCTK corpus.
            subset (str): Subdirectory inside the corpus containing audio (commonly 'wav48').
            transform (callable, optional): An optional transform to be applied on the audio.
        """
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.sr = 16000  # Sample rate for the audio files
        
        self.wavlm_fn = wavlm_fn
        self.wavlm = wavlm
        self.device = device
        # if self.wavlm is not None:
        #     self.wavlm.eval()
        #     for param in self.wavlm.parameters():
        #         param.requires_grad = False
                
        # self.SPEAKER_INFORMATION_LAYER = 6
                
        self.samples = []
        # Each subdirectory in self.wav_dir is assumed to be a speaker folder
        if not os.path.isdir(self.wav_dir):
            raise FileNotFoundError(f"Directory {self.wav_dir} not found. Check your VCTK path.")

        for speaker in sorted(os.listdir(self.wav_dir)):
            speaker_path = os.path.join(self.wav_dir, speaker)
            if os.path.isdir(speaker_path):
                for file in sorted(os.listdir(speaker_path)):
                    if file.lower().endswith('.wav'):
                        file_path = os.path.join(speaker_path, file)
                        self.samples.append((speaker, file_path))
                        
        print(f"Loaded {len(self.samples)} audio files from {len(os.listdir(self.wav_dir))} speakers.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        speaker, file_path = self.samples[idx]
        wav, sr = torchaudio.load(file_path, normalize=True)
        
        # resample to self.sr if necessary
        if not sr == self.sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sr)
            sr = self.sr
                
        wav = wav.to(self.device)
        
        if self.transform:
            wav = self.transform(wav)
            
        with torch.no_grad():
            wav_embs = self.wavlm_fn(self.wavlm, wav)

        return {
            "speaker": speaker,
            "wavlm_embeds": wav_embs,
            "sample_rate": sr,
            "file_name": Path(file_path).stem
        }

class HallucinatorSetDataset(Dataset):
    def __init__(self, dataset: Dataset, hall_dataset_path: str, exists_hall_dataset: bool=True) -> None:
        super().__init__()
        
        if not exists_hall_dataset:
            # assert os.path.isdir(hall_dataset_path), f"Path {hall_dataset_path} should be a directory"
            
            self.save_dir = Path(hall_dataset_path)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            self.dataset = dataset
            
            self.dataset_info = []
            
            print("Saving embeddings with metadata to disk:")
            for idx in tqdm(range(len(dataset))):
                sample = dataset[idx]
                speaker = sample["speaker"]
                file_name = sample["file_name"]
                wavlm_embeds = sample["wavlm_embeds"]
                sample_rate = sample["sample_rate"]
                
                to_save = {
                    "speaker": speaker,
                    "file_name": file_name,
                    "sample_rate": sample_rate,
                    "wavlm_embeds": wavlm_embeds,
                }
                
                embed_file = f"{file_name}.pt"
                save_path = self.save_dir / embed_file
                
                torch.save(to_save, save_path)
                
                self.dataset_info.append(save_path)
        else:
            assert os.path.exists(hall_dataset_path), f"Path {hall_dataset_path} should be a directory"
            self.save_dir = Path(hall_dataset_path)
            self.dataset_info = list(self.save_dir.glob("*.pt"))
        
        self.num_gen_samples_per_idx = [random.randint(1, 5) for idx in range(len(self.dataset_info))]
        
        self.pre_num_gen_samples_per_idx = np.cumsum(self.num_gen_samples_per_idx)
        
        self.MIN_MASKED = 1
        self.MAX_MASKED_PAD = 5
        
    def __len__(self) -> int:
        return sum(self.num_gen_samples_per_idx)
    def __getitem__(self, idx: int) -> dict:
        i = np.searchsorted(self.pre_num_gen_samples_per_idx, idx, side='left')
        
        embeds = torch.load(self.dataset_info[i])
        speaker = embeds["speaker"]
        file_name = embeds["file_name"]
        sample_rate = embeds["sample_rate"]
        
        N, _ = embeds["wavlm_embeds"].shape

        permuted_idx = torch.randperm(N)

        sample_idx = permuted_idx[:200]

        wavlm_embeds = embeds["wavlm_embeds"][sample_idx, :]
        
        
        # # randomly permute wavlm_embeds along dim=1 using
        wavlm_embeds = wavlm_embeds[torch.randperm(wavlm_embeds.shape[0]), :]

        
        n_masked = random.randint(self.MIN_MASKED, wavlm_embeds.shape[0] - self.MAX_MASKED_PAD)
        
        
        # Randomly select indices to mask
        mask_indices = random.sample(range(wavlm_embeds.shape[0]), n_masked)
        mask_indices = sorted(mask_indices)
        mask_indices = torch.tensor(mask_indices)
        
        
        # Create a mask tensor
        mask_tensor = torch.ones(wavlm_embeds.shape[0], dtype=torch.bool)
        
        mask_tensor[mask_indices] = False
        
        # create another tensor that basically has all the masked embeddings
        masked_embeds = wavlm_embeds[~mask_tensor]
        
        # # create another tensor that has all the unmasked embeddings
        # unmasked_embeds = wavlm_embeds[~mask_tensor]
        
        return{
            "masked_out_set": masked_embeds,
            "total_speech_set": wavlm_embeds,
            "mask": mask_tensor 
        }
        
        
def hallu_collate_fn(batch):
    # Unzip the batch
    masked_out_set = [item["masked_out_set"] for item in batch]
    total_speech_set = [item["total_speech_set"] for item in batch]
    mask = [item["mask"] for item in batch]
    
    # Stack the tensors using nested tensors
    masked_out_set = torch.nested.nested_tensor(masked_out_set, dtype=masked_out_set[0].dtype, layout=torch.jagged)
    total_speech_set = torch.nested.nested_tensor(total_speech_set, dtype=total_speech_set[0].dtype, layout=torch.jagged)
    mask = torch.nested.nested_tensor(mask, dtype=mask[0].dtype, layout=torch.jagged)
    
    return {
        "masked_out_set": masked_out_set,
        "total_speech_set": total_speech_set,
        "mask": mask
    }

def get_datatset(path: str='.', url:str = None):
    def get_dataset_curl(path: str='.', url:str = None):
        assert url is not None, "url must be specified"
        # assert os.path.exists(path), f"Path {path} does not exist"
        
        subprocess.run(["curl", "-L", "-o", path, url])
        
    #get_dataset_curl(path, url)
    
    def get_dataset_gdown(path: str='.', url_id:str = None):
        
        assert url is not None, "url must be specified"
        assert os.path.isdir(path), f"Path {path} should be a zip file not a directory"
        
        base_url = "https://drive.google.com/uc?export=download&id="
        
        url = base_url + url_id
        
        gdown.download(url, path, quiet=False)
        
    get_dataset_gdown(path, url)
    
    
def unzipy(path: str):
    base_name = os.path.splitext(path)[-2].split('/')[-1]
    
    print(f"Unzipping {path} ...")
    
    result = subprocess.run(
        ["unzip", "-o", path, "-d", Path(path).parent.absolute()],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    subprocess.run(
        ["rm", path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result
    
    
        
if __name__ == "__main__":
    # vctk_root = "./data/VCTK-Corpus-0.92"
    
    # dataset = VCTKDataset(root_dir=vctk_root, subset="wav48", transform=None)
    
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    
    # for batch in dataloader:
    #     speakers = batch["speaker"]
    #     waveforms = batch["waveform"]
    #     sample_rates = batch["sample_rate"]
    #     print(f"Batch speakers: {speakers}")
    #     print(f"Waveform shape: {waveforms.shape}")
    #     break
    # get_datatset(path=os.path.join(DATASET_DIR, 'vctk-corpus'), url="https://www.kaggle.com/api/v1/datasets/download/kynthesis/vctk-corpus")
    # unzipy(os.path.join(DATASET_DIR, 'vctk-corpus.zip'))
    pass

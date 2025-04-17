from .wavlm import wavlm_large
from .hifigan import hifigan_wavlm
from .knn import KNeighborsVC
from tqdm import tqdm

import os
import gdown

import torchaudio
from .hallucinator.data import VCTKDataset, HallucinatorSetDataset
from .hallucinator import SetDDPM, cosine_beta_schedule
import torch
from torch.nested import nested_tensor

from .phantom import PhantomTransformer, ready_made_model

from .knn import vc

from .utils import wavlm_embedding, wavlm_func_gen
from warnings import filterwarnings
filterwarnings("ignore")

def vc_demo(src_path, dest_path, out_path):
    set_hallucinator = SetDDPM(
        T_timesteps=255, 
        schedule=cosine_beta_schedule,
        device=torch.device('cuda:0')
    )
    file_ids = [
        "1A4WiJ27Q1tBfUFKLAb9jNddiYHmtLEfn",
    ]
    
    # Desired output file names
    file_names = ["halu_model.pt"]
    
    # Google Drive direct download URL base format
    base_url = "https://drive.google.com/uc?export=download&id="
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    download_dir = os.path.join(current_dir, "hallucinator", "checkpoints")
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    url = base_url + file_ids[0]
    output_path = os.path.join(download_dir, file_names[0])
    
    print(f"Downloading {file_names[0]} from {url}...")
    gdown.download(url, output_path, quiet=False)
    print(f"Downloaded {file_names[0]} to {output_path}")
    
    ph = ready_made_model()
    
    vc_model = vc(hallucinator=set_hallucinator, transf=ph)
    
    src_seq = vc_model.get_features(src_path)
    
    target_seq = vc_model.get_matching_set(dest_path)
    
    out_wav =vc_model.match(src_seq, target_seq, topk=4)
    
    torchaudio.save(out_path, out_wav, 16000)
if __name__ == "__main__":
    
    set_hallucinator = SetDDPM(
        T_timesteps=255, 
        schedule=cosine_beta_schedule,
        device=torch.device('cuda:0')
    )
    
    src_path = ''
    dest_path = ''
    
    out_path = ''
    
    ph = ready_made_model()
    
    vc_model = vc(hallucinator=set_hallucinator, transf=ph)
    
    src_seq = vc_model.get_features(src_path)
    
    target_seq = vc_model.get_matching_set(dest_path)
    
    out_wav =vc_model.match(src_seq, target_seq, topk=4)
    
    torchaudio.save(out_path, out_wav, 16000)
    
    
    
    
    
    
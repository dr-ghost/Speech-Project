from wavlm import wavlm_large
from hifigan import hifigan_wavlm
from knn import KNeighborsVC, knn_vc
from tqdm import tqdm

from hallucinator.data import VCTKDataset, HallucinatorSetDataset
from hallucinator import SetDDPM, cosine_beta_schedule
import torch
from torch.nested import nested_tensor

from utils import wavlm_embedding, wavlm_func_gen
from warnings import filterwarnings
filterwarnings("ignore")

if __name__ == "__main__":
    
    wavlm_model = wavlm_large()
    wavlm_model.eval()
    for param in wavlm_model.parameters():
        param.requires_grad = False
        
    wavlm_model = torch.compile(wavlm_model)
    
    vctk_dat = VCTKDataset(root_dir="models/hallucinator/dataset/VCTK-Corpus", wavlm_fn=wavlm_embedding, wavlm=wavlm_model, device=torch.device("cuda"))
    # print(len(vctk_dat))
    
    # min = 29 max = 963 seqs
    
    hallu_dat = HallucinatorSetDataset(None, hall_dataset_path="models/hallucinator/dataset/Hallu-Corpus", exists_hall_dataset=True)
    
    set_ddpm = SetDDPM(
        T_timesteps=255,
        schedule=cosine_beta_schedule,
        device=torch.device("cuda")
    )
    
    dct = hallu_dat[0]
    
    
    # set_ddpm.sample_batch(64, [200 for i in range(64)], nested_tensor([dct['total_speech_set']]), nested_tensor([dct['mask']]))
    
    set_ddpm.train(20, hallu_dat, batch_size=64, model_save_path="models/hallucinator/checkpoints/test_1_hallu.pt")
    
    
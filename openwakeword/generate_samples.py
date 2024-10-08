from melo.api import TTS
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from random import randint
from tqdm import trange

def generate_samples(
    text: Union[List[str], str],
    output_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    file_names: Optional[List[str]] = None,
    model: Union[str, Path] = "",
    batch_size: int = 1,
    slerp_weights: Tuple[float, ...] = (0.5,),
    length_scales: Tuple[float, ...] = (0.75, 1, 1.25),
    noise_scales: Tuple[float, ...] = (0.667,),
    noise_scale_ws: Tuple[float, ...] = (0.8,),
    max_speakers: Optional[float] = None,
    verbose: bool = False,
    auto_reduce_batch_size: bool = False,
    min_phoneme_count: Optional[int] = None,
    **kwargs,
) -> None:
    device = 'auto' # or cuda:0
    model = TTS(language='ZH', device=device)
    speed = 1.0
    speaker_ids = model.hps.data.spk2id
    for _ in trange(max_samples):
        wav_path = output_dir / f"{randint(100000000,1000000000)}.wav"
        model.tts_to_file(
            text,
            speaker_ids['ZH'],
            wav_path,
            noise_scale=noise_scales[0],
            noise_scale_w=noise_scale_ws[0],
            speed=speed
        )
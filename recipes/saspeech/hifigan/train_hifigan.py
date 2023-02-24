import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = os.path.dirname(os.path.abspath(__file__))

audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample = True,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = HifiganConfig(
    batch_size=32,
    audio=audio_config,
    eval_batch_size=16,
    num_loader_workers=20,
    num_eval_loader_workers=14,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=os.path.join(output_path, "../saspeech_raw_wav_large/"),
    output_path=output_path,
    dashboard_logger = 'wandb',
)

# init audio processor
ap = AudioProcessor.init_from_config(config)

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()

import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.overflow_config import OverflowConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.overflow import Overflow
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from hebrew import Hebrew, chars

output_path = os.path.dirname(os.path.abspath(__file__))

# init configs
dataset_config = BaseDatasetConfig(
    #formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join("data", "saspeech_gold_clean/")
    formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join("data", "saspeech_gold_nikud/")
)

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

config = OverflowConfig(  # This is the config that is saved for the future use
    run_name="overflow_saspeech_gold_nikud",
    audio=audio_config,
    batch_size=10,
    shuffle=True,
    drop_last=True,
    #start_by_longest=True,
    eval_batch_size=16,
    num_loader_workers=6,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    # phoneme_language="en-us",
    # phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    precompute_num_workers=8,
    mel_statistics_parameter_path=os.path.join(output_path, "sa_parameters.pt"),
    force_generate_statistics=False,
    print_step=1,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    characters = CharactersConfig(
        characters="אבגדהוזחטיכךלמםנןסעפףצץקרשת0123456789।%$₪ '\"-!?,.…" + ''.join(i.char for i in chars.NIQQUD_CHARS),
        punctuations="–_:;‘’“”()",
        is_unique=True
    ),
    lr=5e-4,
    test_sentences=['עכשיו, לאט לאט, נסו לדמיין סופרמרקט.'],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# make sure that "achshav, le'at le'at, nasu ledamien supermarket" sentence is not in
# the training set by forcibly moving it into the evaluation set
for i, item in enumerate(train_samples):
    if 'עכשיו, לאט לאט, נסו לדמיין סופרמרקט' in str(Hebrew(item['text']).text_only()):
        print(f'Found eval sentence in training samples (index {i}), moving to evaluation set')
        eval_samples.append(item)
        train_samples.pop(i)
        break

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = Overflow(config, ap, tokenizer)


# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    gpu=1,
)
trainer.fit()

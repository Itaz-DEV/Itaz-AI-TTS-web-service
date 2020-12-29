from source.text.symbols import symbols

class Hparams():
        ################################
        # Experiment Parameters        #
        ################################
        epochs=5000
        iters_per_checkpoint=5000
        seed=1234
        dynamic_loss_scaling=True
        fp16_run=True
        distributed_run=False
        dist_backend="nccl"
        dist_url="tcp://localhost:16029"
        cudnn_enabled=True
        cudnn_benchmark=False
        ignore_layers=['embedding.weight']

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False
        training_files='filelists/ljs_audio_text_train_filelist.txt'
        validation_files='filelists/ljs_audio_text_val_filelist.txt'
        text_cleaners=['english_cleaners']

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0
        sampling_rate=22050
        filter_length=1024
        hop_length=256
        win_length=1024
        n_mel_channels=80
        mel_fmin=0.0
        mel_fmax=8000.0

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols)
        symbols_embedding_dim=512

        ###  jeju
        #       male
        t_path_jeju_male="source/outdir/male/jeju/checkpoint_70056"
        w_jeju_path_male="source/outdir/male/gyeongsang/waveglow_gyeongsang_266000"
        #       female
        t_path_jeju_female=""
        w_jeju_path_female=""

        ### standard
        #       female
        t_path_standard_female="source/outdir/female/standard/t_kss_206000_best"
        w_jeju_standard_female="source/outdir/female/standard/waveglow_kss_315000"
        #       male
        t_path_standard_male=""
        w_jeju_standard_male=""


        ### gyeongsang
        #       male
        t_path_gyeongsang_male="source/outdir/male/gyeongsang/checkpoint_58000"
        w_gyeongsang_path_male="source/outdir/male/gyeongsang/waveglow_gyeongsang_266000"

        #       female
        t_path_gyeongsang_female=""
        w_gyeongsang_path_female=""


        ###jeonla
        # female
        t_path_jeon_female="source/outdir/female/jeon/checkpoint_155000"
        w_jeon_path_female="source/outdir/female/jeon/waveglow_240000"
        ###jeonla
        # male
        t_path_jeon_male=""
        w_jeon_path_male=""
        ###

        ### yeonbyeon
        #       male
        t_path_yeonbyeon_male = "source/outdir/female/yeonbyeon/checkpoint_6000_keep_for_backup"
        w_yeonbyeon_path_male = "source/outdir/male/gyeongsang/waveglow_gyeongsang_266000"
        ####       female
        t_path_yeonbyeon_female = ""
        w_yeonbyeon_path_female = ""
        ###



        # Encoder parameters
        encoder_kernel_size=5
        encoder_n_convolutions=3
        encoder_embedding_dim=512

        # Decoder parameters
        n_frames_per_step=1  # currently only 1 is supported
        decoder_rnn_dim=1024
        prenet_dim=256
        max_decoder_steps=4000
        gate_threshold=0.5
        p_attention_dropout=0.1
        p_decoder_dropout=0.1

        # Attention parameters
        attention_rnn_dim=1024
        attention_dim=128

        # Location Layer parameters
        attention_location_n_filters=32
        attention_location_kernel_size=31

        # Mel-post processing network parameters
        postnet_embedding_dim=512
        postnet_kernel_size=5
        postnet_n_convolutions=5

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False
        learning_rate=1e-3
        weight_decay=1e-6
        grad_clip_thresh=1.0
        batch_size=32
        mask_padding=True  # set model's padded outputs to padded values

def create_hparams():
    """Create model hyperparameters. Parse nondefault from given string."""
    hparams = Hparams()
    return hparams


class CONFIG:
    gpus = "0"  # List of gpu devices

    class TRAIN:
        sainty_check = True  # if True will try to remember <sainty_size> samples, from SaintyCheckLoader
        sainty_size = 1 # number of samples to remember
        batch_size = 96  # number of audio files per batch
        lr = 1e-2  # learning rate
        limit_val_batches = 1
        epochs = 1000  # max training epochs
        check_val_every_n_epoch = 10 # run validation each
        workers = 1  # number of dataloader workers
        val_split = 0.02  # validation set proportion
        clipping_val = 1.0  # gradient clipping value
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor

    # Model config
    class MODEL:
        enc_layers = 4  # number of MLP blocks in the encoder
        enc_in_dim = 384  # dimension of the input projection layer in the encoder
        enc_dim = 768  # dimension of the MLP blocks
        pred_dim = 512  # dimension of the LSTM in the predictor
        pred_layers = 1  # number of LSTM layers in the predictor

    # Dataset config
    class DATA:
        dataset = 'dns_fullband'  # dataset to use
        '''
        Dictionary that specifies paths to root directories and train/test text files of each datasets.
        'root' is the path to the dataset and each line of the train.txt/test.txt files should contains the path to an
        audio file from 'root'. 
        '''
        data_dir = {'vctk': {'root': 'data/vctk/wav48',
                             'train': "data/vctk/train.txt",
                             'test': "data/vctk/test.txt"},

                    'dns_fullband': {'root': 'data/dns_fullband/',
                                     'train': "data/large_fullband_train.txt",
                                     'test': "data/large_fullband_test.txt"},
                    }

        assert dataset in data_dir.keys(), 'Unknown dataset.'
        sr = 48000  # audio sampling rate
        audio_chunk_len = 122880  # size of chunk taken in each audio files
        window_size = 960  # window size of the STFT operation, equivalent to packet size
        stride = 480  # stride of the STFT operation

        class TRAIN:
            packet_sizes = [960]  # 256, 512, 768, 960, 1024, 1536 packet sizes for training. All sizes should be divisible by 'audio_chunk_len'
            transition_probs = ((0.9, 0.1), (0.5, 0.1), (0.5, 0.5))  # list of trainsition probs for Markow Chain

        class EVAL:
            packet_size = 960  # 20ms
            transition_probs = [(0.9, 0.1)]  # (0.9, 0.1) ~ 10%; (0.8, 0.2) ~ 20%; (0.6, 0.4) ~ 40%
            masking = 'gen'  # whether using simulation or real traces from Microsoft to generate masks
            assert masking in ['gen', 'real']
            trace_path = 'test_samples/blind/lossy_singals'  # must be clarified if masking = 'real'

    class LOG:
        log_dir = 'cosine_sainty_lightning_logs'  # checkpoint and log directory
        sample_path = 'cosine_sainty_audio_samples'  # path to save generated audio samples in evaluation.

    class TEST:
        in_dir = 'blind/lossy_signals'  # path to test audio inputs
        real_dir = 'X_CleanReference'
        out_dir = 'blind/lossy_signals_out'  # path to generated outputs
        out_dir_orig = 'blind/lossy_signals48k'

    class NBTEST:
        to_synthesize = 1 # first n samples from real_dir will be synthesized with the loss in loss_path
        packet_size = 960  # 20ms
        repeat_factor = 3 # will load real mask for orig sample rate, how much should repeat each sample (diff target sr / orig sr)
        transition_probs = [(0.9, 0.1)]  # (0.9, 0.1) ~ 10%; (0.8, 0.2) ~ 20%; (0.6, 0.4) ~ 40%
        masking = 'real'  # whether using simulation or real traces from Microsoft to generate masks
        assert masking in ['gen', 'real']
        loss_path = 'blind/lossy_signals'  # must be clarified if masking = 'real'
        real_dir = 'X_CleanReference'
        out_dir = 'causal_training/gen'  # path to generated outputs
        out_dir_orig = 'causal_training/loosy'

    class WANDB:
        project = "FRN"
        log_n_audios = 1
        monitor = "train_stft_loss"



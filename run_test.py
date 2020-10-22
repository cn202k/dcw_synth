from training import Training_autoencoder

training = Training_autoencoder()

training.dataset_location = './dataset'
training.ts_log_dir = './tensorboard_logs'
training.experiment_name = 'test'
training.model_save_dir = './saved_models/%s' % training.experiment_name
training.version = 0
training.test_run = True
training.num_workers = 1
training.gpus = None

training.batch_size = 16
training.max_epochs = 2
training.lr = 1e-3
training.lr_decay = None
training.fine_tune = False

training.start()

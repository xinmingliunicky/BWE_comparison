import os
import argparse
import datetime
import torch

# Control the task to be performed in commandline by adding different options

parser = argparse.ArgumentParser()  # provide a commandline option for choosing tasks
parser.add_argument('--model', type=str, default='unet', help='Architecture: unet or resnet')  # 'unet' or 'resnet'
# pretrained models to choose from: unet, unetda, unetbn, unetdo, resnet, resnetda, resnetbn, resnetdo
parser.add_argument('--load', type=str, default=None,
                    help='Loads pretrained model. Make sure arguments (model, batchnorm, dropout) are set accordingly')
parser.add_argument('--multifilter', action="store_true", 
                    help="Activates data augmentation using multiple low-pass filters")  # --multifilter will use data augmentation
parser.add_argument('--batchnorm', action="store_true", help="Batch normalization")  # --batchnorm will use batch normalization
parser.add_argument('--dropout', action="store_true", help="Dropout")  # --dropout will use dropout
parser.add_argument('--test', action="store_true", help="Test only, no training")  # --test will ONLY run testing
parser.add_argument('--batchsize', type=int, default=8, help="Batch size")
parser.add_argument('--n_workers', type=int, default=8, help='Number of cpu cores to use')
parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
args = parser.parse_args()  # args stores the above options, args.XX activates them

# below use CONSTANTS to hold tasks for execution (run.py)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # automatically use GPU when available

TEST_ONLY = args.test

DROPOUT = 0.5 if args.dropout else 0.0

# the last 8 songs of DSD100 training set is used for validation
N_SONGS_VALID = 8     # number of songs for validation
DURATION_VALID = 8     # duration in seconds for validation (seconds)
START_VALID = 8     # start of validation (seconds)

METRIC_TRAIN = 'trn_snr'    # metric to adjust lr

N_SONGS_TEST = None
DURATION_TEST = None

LOAD_MODEL = args.load
MODEL = args.model  # 'unet.pt', 'unetda.pt',...

MULTIFILTER = args.multifilter

if args.lr > 0:  # if learning rate specified, use it
    LEARNING_RATE = args.lr
elif 'resnet' in MODEL:  # default learning rate for resnet, resnetda, resnetbn, resnetdo
    LEARNING_RATE = 1e-5
else:
    LEARNING_RATE = 5e-5  # default learning rate for unet, unetda, unetbn, unetdo

OVERWRITE_LR = False    # overwrites learning rate if model is loaded

BATCHNORM = args.batchnorm
BATCH_SIZE = args.batchsize

EXPERIMENT = MODEL
# MODEL is baseline, EXPERIMENT is MODEL with a normalization method enabled
if BATCHNORM:
    EXPERIMENT += '_bn'
if DROPOUT > 0:
    EXPERIMENT += '_do'
if MULTIFILTER:
    EXPERIMENT += '_da'
# thus EXPERIMENT could be unet/unetda/....

ITER_VAL = 2500   # Tests, record loss, saves models, samples every ____ iterations
VALID = True    # perform validation or not
# To adjust learning rate
PATIENCE = int(15 * 2500 / ITER_VAL)
LR_FACTOR = 0.1

# LPF to use under different cases: see TABLE 1
if MULTIFILTER:  # training with data augmentation
    FILTERS_TRAIN = [
        ('cheby1', 6), ('cheby1', 8),
        ('cheby1', 10), ('cheby1', 12),
        ('bessel', 6), ('bessel', 12),
        ('ellip', 6), ('ellip', 12)
    ]
else:  # training without data augmentation
    FILTERS_TRAIN = [('cheby1', 6)]
# By default, validation is done using training (seen) filters, so no need to specify again
# when need to perform validation with unseen filters:
FILTERS_VALID = [('butter', 6)]
FILTERS_TEST = [('cheby1', 6), ('butter', 6)]
CUTOFF = 11025  # sample rate=44100, nyquist freq=22050, cutoff=1/2*nyq: see Figure 2

SAMPLE_RATE = 44100

SAMPLE_LEN = 2**13   # length of training samples

# parameters for outputting wav files
WAV_SAMPLE_LEN = 2**13    # if not float, it is a ratio
WAV_BATCH_SIZE = BATCH_SIZE
TEST_DURATION = None    # None for entire song

PAD_TYPE = 'zero'  # padding

L_LOSS = 2      # L1 or L2 loss

MAX_ITER = 500000  # Training ends after this many iterations
MIN_LR = 1e-8   # Learning rate halves when loss reaches plateau, training ends after learning rate smaller than this

NUM_WORKERS = args.n_workers     # Number of CPU cores for loading and processing batches

ADAPTIVE_LR = True

SAVE_MODEL = True

assert LOAD_MODEL or not TEST_ONLY   # if you're testing, you need to load a model

# save your model after training or testing, saved model will have no extension
DATE = datetime.datetime.now().strftime("%m-%d")
if TEST_ONLY:
    SAVE_NAME = LOAD_MODEL.replace('.pt', '')  # e.g. unetda
else:
    SAVE_NAME = DATE + '_' + EXPERIMENT  # e.g. MM-DD-unet_da

MAIN_DIR = os.path.abspath('..')

SAVE_SAMPLES = True     # saves an entire song created by the model

OUTPUT_DIR = os.path.join(MAIN_DIR, 'output')

TRAIN_DIRS = [
    # os.path.join(MAIN_DIR, 'datasets', 'medleydb'),  # MedlyDB dataset requires permission to download
    os.path.join(MAIN_DIR, 'datasets', 'DSD100', 'Mixtures', 'Dev')
]
TEST_DIRS = [os.path.join(MAIN_DIR, 'datasets', 'DSD100', 'Mixtures', 'Test')]

MODEL_DIR = os.path.join(MAIN_DIR, OUTPUT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
GENERATION_DIR = os.path.join(MAIN_DIR, OUTPUT_DIR, 'generation', SAVE_NAME)
os.makedirs(GENERATION_DIR, exist_ok=True)

# if using multiple filters, you need that many songs for validation, so that it is one filter per song (each filter has equal weight in the average)
assert len(FILTERS_TRAIN) == 1 or len(FILTERS_TRAIN) == N_SONGS_VALID or TEST_ONLY

from easydict import EasyDict as edict

#######################
# paths
#######################
DATA_DIR = '/media/storage/Kaggle/train/data_1'
MODEL_SAVE_DIR = './models'
LOAD_MODEL = False

#######################
# training parameters:
#######################
TRAIN = 'rpn'  # 'rpn' or 'mask'
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MODEL_SAVE_PERIOD = 500  # steps after which to save model
DISPLAY_PERIOD = 10  # steps after which to display losses

#######################
# Data parameters:
#######################
INPUT_SHAPE = (128, 128, 3)
IMG_MEAN = [0.67112247, 0.57436889, 0.78549767]
IMG_STD = [0.13055238, 0.20516671, 0.065434]

#######################
# Network parameters:
#######################
N_BLOCK_LAYERS = 3  # number of up/down sampling block layers
DIR_BINS = 16  # number of direction bins in segmentation mask

#######################
# Densenet parameters:
#######################
BATCH_NORM = False
DROP_RATE = 0.2  # drop
# number of layers within each block layer (key is the block layer)
LAYERS_PER_BLOCK = {'0': 4, '1': 5, '2': 7, '3': 10}

#######################
# rcnn parameters:
#######################
# we keep same ratios for all block layers
ANCHOR_RATIOS = [(1, 1), (1, 2), (2, 1)]
# anchor scales for each block layer (key is the block layer)
ANCHOR_SCALES = {'1': [4], '2': [8, 20], '3': [40, 80]}
SCALE_RANGE = {'1': (3, 8), '2': (6, 40), '3': (36, 128)}
IOU_LOW = 0.3
IOU_HIGH = 0.7
# class reweighting factor for balancing classes
CLASS_BALANCE = 0.5  # between 0 and 1;  0 means no reweighting
REG_WEIGHT = 0.2  # weight for regression loss
SEM_SEG_WEIGHT = 0.2  # weight for semantic segmentation loss
DIR_SEG_WEIGHT = 0.2  # weight for direction segmentation loss
FEAT_CROP_SIZE = [8, 8]  # feature size for RCN stage

##############################################
cfg = edict({'data_dir': DATA_DIR,
             'model_save_dir': MODEL_SAVE_DIR,
             'load_model': LOAD_MODEL,
             'train': TRAIN,
             'epochs': EPOCHS,
             'batch_size': BATCH_SIZE,
             'learning_rate': LEARNING_RATE,
             'save_period': MODEL_SAVE_PERIOD,
             'display_period': DISPLAY_PERIOD,
             'input_shape': INPUT_SHAPE,
             'image_mean': IMG_MEAN,
             'image_stddev': IMG_STD,
             'batch_norm': BATCH_NORM,
             'drop_rate': DROP_RATE,
             'n_block_layers': N_BLOCK_LAYERS,
             'dir_bins': DIR_BINS,
             'layers_per_block': LAYERS_PER_BLOCK,
             'anchor_scales': ANCHOR_SCALES,
             'anchor_ratios': ANCHOR_RATIOS,
             'scale_range': SCALE_RANGE,
             'iou_low': IOU_LOW,
             'iou_high': IOU_HIGH,
             'class_balance': CLASS_BALANCE,
             'reg_weight': REG_WEIGHT,
             'sem_seg_weight': SEM_SEG_WEIGHT,
             'dir_seg_weight': DIR_SEG_WEIGHT,
             'feat_crop_size': FEAT_CROP_SIZE})

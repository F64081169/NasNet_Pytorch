# NasNet_Pytorch

Using pre-trained nasnet model inference a photo via pytorch

## Usage 

```
mkdir data
```
Next, put the photo you want to predict with under the /data directory.
```
pip install pretrainedmodels 
```
```
python nasnet_test.py
```

## Configuration(nasnet_config.py)

```
# Configuration
num_processes = 3
cores_per_process = 3
batch_size = 100
eval_image_size = 236
num_test_data_per_process = 200
dataset_name = 'imagenet'
num_preprocessing_threads = 4
labels_offset = 0
model_name = 'inceptionv3'
preprocessing_name = 'inceptionv3'
moving_average_decay = None
quantize = False
use_grayscale = False
```

reference:https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/nasnet.py?fbclid=IwAR3flVccOmWswIQPbMQB5CjQItRJHNCL6kxpb1UtiTim1_OX8St8lPbNNAM

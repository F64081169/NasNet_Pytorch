import torch
from torch.autograd import Variable
import pretrainedmodels as pm
import pretrainedmodels.utils as utils
import ssl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

img_path = 'data/cat_224.jpg'

# Configuration
num_processes = 3
batch_size = 100
eval_image_size = 224
num_test_data_per_process = 200
dataset_name = 'imagenet'
# num_preprocessing_threads = 4
# labels_offset = 0
# model_name = 'inceptionv3'
# moving_average_decay = None
# quantize = False
# use_grayscale = False

class CustomDataset(Dataset):
    def __init__(self, img_path, transform=None, num_samples=1):
        self.img_paths = [img_path] * num_samples
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img

# Transform the input image
# transform = transforms.Compose([
#     transforms.Resize((eval_image_size, eval_image_size)),
#     transforms.ToTensor(),
# ])

# Load the NASNet model
nasnet_model = pm.inceptionv3(num_classes=1000, pretrained=dataset_name)
nasnet_model.eval()
transform = utils.TransformImage(nasnet_model)
# Create a DataLoader for parallel processing
dataset = CustomDataset(img_path, transform=transform, num_samples=num_test_data_per_process)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_processes)

for batch in dataloader:
    x = Variable(batch, requires_grad=False)

    # Get the output logits
    out_logits = nasnet_model(x)

    print("Shape of output logits:", out_logits.shape)

    probs = F.softmax(out_logits, dim=1)

    predicted_class_indices = torch.argmax(probs, dim=1)

    print("Predicted Probabilities:", probs)
    print("Predicted Class Indices:", predicted_class_indices.tolist(), "Batch Size:", len(predicted_class_indices))

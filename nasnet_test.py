import torch
from torch.autograd import Variable
import pretrainedmodels as pm
import pretrainedmodels.utils as utils
import ssl
import torch.nn.functional as F

ssl._create_default_https_context = ssl._create_unverified_context

# Load an image for testing
img = utils.LoadImage()('data/cat.jpg')

# Choose the NASNet model and pretrained setting
model_name = 'nasnetalarge'
pretrained_setting = 'imagenet'

# Load the NASNet model
nasnet_model = pm.nasnetalarge(num_classes=1000, pretrained=pretrained_setting)
nasnet_model.eval()

# Transform the input image
transformed_img = utils.TransformImage(nasnet_model)(img)
transformed_img = transformed_img.unsqueeze(0)
x = Variable(transformed_img, requires_grad=False)

# Get the output logits
out_logits = nasnet_model(x)

print("Shape of output logits:", out_logits.shape)

probs = F.softmax(out_logits, dim=1)

predicted_class_index = torch.argmax(probs, dim=1).item()

print("Predicted Probabilities:", probs)

print("Predicted Class Index:", predicted_class_index)

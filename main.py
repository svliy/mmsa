import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), './pt/model_weights.pth')
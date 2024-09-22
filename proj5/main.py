'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
# step 1
import numpy as np
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform


# step 2
transform = get_transform(image_size=384)
model = ram_plus(pretrained=pretrained,
                             image_size=384,
                             vit='swin_l')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# step 3

image = transform(Image.open('images')).unsqueeze(0).to(device)

# step 4
res = inference(image, model)

# step5
print("Image Tags: ", res[0])


    

    

    

    

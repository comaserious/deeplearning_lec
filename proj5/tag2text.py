'''
 * The Tag2Text Model
 * Written by Xinyu Huang
'''

# step 1
import numpy as np
import random

import torch

from PIL import Image
from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform

# step 2~3

# step 2 instance inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = get_transform(image_size=384)
delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]
model = tag2text(pretrained=args.pretrained,
                             image_size=384,
                             vit='swin_b',
                             delete_tag_index=delete_tag_index)

model.threshold = 0.68  # threshold for tagging
model.eval()
model = model.to(device)

# step 3

image = transform(Image.open('images\Vu2Nqwb.jpeg')).unsqueeze(0).to(device)


    
# step 2~3

# step 4
res = inference(image, model)

# step 5
print("Model Identified Tags: ", res[0])
print("User Specified Tags: ", res[1])
print("Image Caption: ", res[2])
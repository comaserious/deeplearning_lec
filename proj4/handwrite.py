# step 1
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import requests 
import unicodedata
from io import BytesIO
from PIL import Image

# step2
processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr") 
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

# step3
url = "https://img1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/K0c/image/dPtLLZCak7u-KDZissxzKROBkns.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# step4
pixel_values = processor(img, return_tensors="pt").pixel_values 
generated_ids = model.generate(pixel_values, max_length=64)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
generated_text = unicodedata.normalize("NFC", generated_text)

# step5
print(generated_text)

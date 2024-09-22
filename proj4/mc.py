# pipeline 을 사용하지 않을때 이런식으로 사용한다

# step 1
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice
import torch

# step 2
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_swag_model")
model = AutoModelForMultipleChoice.from_pretrained("stevhliu/my_awesome_swag_model")

# step 3
prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law does not apply to croissants and brioche."
candidate2 = "The law applies to baguettes."

# step 4
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0)

outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
logits = outputs.logits
result = logits.argmax().item()


# step 5
print(result)
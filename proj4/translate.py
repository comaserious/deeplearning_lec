# step 1
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# step 2
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

# step 3
text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."

inputs = tokenizer(text, return_tensors="pt").input_ids

# step 4

outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)



tokenizer.decode(outputs[0], skip_special_tokens=True)


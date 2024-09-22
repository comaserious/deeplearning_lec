# step 1 module
from transformers import pipeline

# step 2 instance inference
question_answerer = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")

# step 3 input data
question = "내 이름은 뭐지"
context = "난 이호준이고 지금 서울에서 살고 있어"

# step 4 inference

result = question_answerer(question=question, context=context)

# step 5
print(result)
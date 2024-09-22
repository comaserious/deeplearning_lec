# step 1
from sentence_transformers import SentenceTransformer

# step 2
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# step 3 
sentence1 = '집에 가자'


sentence2 = '날씨가 좋다'


sentence3 = '선선하네'


# step 4 inference (classify, detect , get , encode ...)
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)

print(embedding1.shape, embedding2.shape , embedding3.shape)
# [3, 384]

# step 5
sim1 = model.similarity(embedding1,embedding2)
sim2 = model.similarity(embedding1,embedding3)
sim3 = model.similarity(embedding2,embedding3)

print(sim1)
print(sim2)
print(sim3)

# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
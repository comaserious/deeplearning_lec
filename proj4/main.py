# step1 . get module
from transformers import pipeline

#step 2 . inference instance
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

#step 3. input data

text = "현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급등"


#step4. inference
result = classifier(text)

#step 5. service
print(result)

if result[0].get('label') == 'positive':
    print('positive sign',result[0]['score'])
else : 
    print ('negative sign',result[0]['score'])


# step 1
import easyocr

# step 2
reader = easyocr.Reader(['ko','en']) 

# step 4
result = reader.readtext('images\\korean.png',detail=0)

# step 5
print(result)
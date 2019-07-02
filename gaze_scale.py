import os
import cv2
'''from pprint import pprint

with open('00-19_30-49.csv', 'r') as f:
    content = f.readlines()

content = [x.strip().split() for x in content]
pprint(content)'''

sizes = set()

allfiles = os.listdir('.')
jpgs = [x for x in allfiles if x[-4:]=='.jpg']

for img in jpgs:
    if img[0]=='2':
        continue
    else:
        #print(img, end='\t\t')
        im = cv2.imread(img)
        #print(im.shape)
        sizes.add(im.shape)

print(sizes)
        

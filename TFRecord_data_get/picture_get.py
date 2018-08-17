from urllib.request import urlretrieve
import os
import pymongo
mongo_url = "10.4.40.129:27017"
client = pymongo.MongoClient(mongo_url)
DATABASE = "dlsdata_v2"
db = client[DATABASE]
COLLECTION = "painting"
db_coll = db[COLLECTION ]
queryArgs = {'infos.text':'肖像画'}
projectionFields = ['image']
searchRes = db_coll.find(queryArgs, projection = projectionFields).limit(2000)
if 'images' not in os.listdir():
    os.makedirs('images')
i=1
for file in searchRes:
    filename=file['image']
    img_url='https://pic.allhistory.com/'+filename
    imagename='image'+str(i)+'.jpg'
    try:
        urlretrieve(url = img_url,filename = 'images/' + imagename)
    except:
        print(img_url)
        continue
    else:
        i = i + 1

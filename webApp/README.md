## WebApp using ReactJS
Potato Disease Classification WebApp

## Install ReactJS
```
Step 1 : sudo apt update

Step 2 : sudo apt upgrade

Step 3 : sudo apt install nodejs

Step 4 : npm install --from-lock-json

Step 5 : npm audit fix
```


## Step 1 :
## start docker container

```
docker run -t --rm -p 8501:8501 -v /media/sujit/3785310C09CB4011/Project/bell-pepper-disease-classification:/bell-pepper-disease-classification tensorflow/serving --rest_api_port=8501 --model_config_file=/bell-pepper-disease-classification/models.config

```

## Step 2 :
## Run FastAPI
```
Step 1 : cd api

Step 2 : pip install -r requirements.txt or pip3 install -r requirements.txt

Step 3 : python main_tf_serving.py.py
```

## Step 3 :
## Run ReactJS

```
Step 1 : cd WebAPP

Step 2 : npm run start
```
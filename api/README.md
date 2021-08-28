## To start docker container

```
docker run -it -v /media/sujit/3785310C09CB4011/Project/bell-pepper-disease-classification:/bell-pepper-disease-classification -p 5000:5000 --entrypoint /bin/bash tensorflow/serving
```

## To serve only latest model

```
tensorflow_model_server --rest_api_port=5000 --model_name=bell_pepper_model --model_base_path=/bell-pepper-disease-classification/saved_models/
```

## To serve models using model config file

```
tensorflow_model_server --rest_api_port=5000  --allow_version_labels_for_unavailable_models --model_config_file=/bell-pepper-disease-classification/models.config
```

## How To Run
```
Step 1 : cd bell-pepper-disease-classification/api

Step 2 : pip install -r requirements.txt or pip3 install -r requirements.txt

Step 3 : python main.py

Step 4 : python main_tf_serving.py.py
```



## TF Serving Installation Instructions & Config File Help
https://www.tensorflow.org/tfx/serving/docker

https://www.tensorflow.org/tfx/serving/serving_config
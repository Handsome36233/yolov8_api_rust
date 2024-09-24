# yolov8_api_rust

rust implementation, using yolov8 combined with actix-web to create a service interface

#### build

```shell
cargo run --release
```

#### run

```shell
./target/release/app onnx_model_path --ip 127.0.0.1 --port 8080
```

example model

https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/yolov8m.onnx

#### request

curl

```shell
curl -X POST http://127.0.0.1:8080/detect \                                                                              🐍 base 18:21:22
     -H "Content-Type: application/json" \
     -d '{"path": "your_image_path"}'
```

python

```python
import requests
url = "http://127.0.0.1:8080/detect"
image_path = "your_image_path"
data = {"path": image_path}
response = requests.post(url, json=data)
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
```

# ml_restapi

$ python3.11 -m venv venv
$ source venv/bin/activate
(venv) $ python -m pip install -r requirements.txt 

(venv) $ fastapi run main.py
INFO     Using path main.py
INFO     Resolved absolute path /tmp/ml_restapi/main.py                                                                                        INFO     Searching for package file structure from directories with __init__.py files                                                          INFO     Importing from /tmp/ml_restapi                                                                                                         ╭─ Python module file ─╮                                                                                                                       │                      │                                                                                                                       │  🐍 main.py          │
 │                      │
 ╰──────────────────────╯

INFO     Importing module main
2024-09-28 05:25:40.890668: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-09-28 05:25:40.894769: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-09-28 05:25:40.909786: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-09-28 05:25:40.940044: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-09-28 05:25:40.946189: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-09-28 05:25:40.969074: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-09-28 05:25:42.289260: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2024-09-28 05:25:43,135 - main - DEBUG - init models dict and models params
2024-09-28 05:25:43,135 - main - DEBUG - starting FASTAPI app main...
INFO     Found importable FastAPI app
 ╭─ Importable FastAPI app ─╮
 │                          │
 │  from main import app    │
 │                          │
 ╰──────────────────────────╯
INFO     Using import string main:app
 ╭─────────── FastAPI CLI - Production mode ───────────╮
 │                                                     │
 │  Serving at: http://0.0.0.0:8000                    │
 │                                                     │
 │  API docs: http://0.0.0.0:8000/docs                 │
 │                                                     │
 │  Running in production mode, for development use:   │
 │                                                     │
 │  fastapi dev                                        │
 │                                                     │
 ╰─────────────────────────────────────────────────────╯

INFO:     Started server process [760168]
INFO:     Waiting for application startup.
2024-09-28 05:25:43,239 - main - DEBUG - lifespan starting...
2024-09-28 05:25:43,239 - main - DEBUG - using saved LE and CLF models
2024-09-28 05:25:43,241 - main - DEBUG - using saved CNN model
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


$ ./curlupload.sh 
retrain_img.zip
{"message":"Data received successfully, model training has started."}


2024-09-28 05:26:29,065 - main - DEBUG - retrain_upload_file starting... file:retrain_img.zip size:833610
INFO:     127.0.0.1:47360 - "POST /upload-images/ HTTP/1.1" 200 OK
Epoch 1/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 2s 106ms/step - accuracy: 0.9926 - loss: 0.0297 - val_accuracy: 0.9800 - val_loss: 0.0565
Epoch 2/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 66ms/step - accuracy: 0.9888 - loss: 0.0330 - val_accuracy: 0.9800 - val_loss: 0.0512
Epoch 3/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 70ms/step - accuracy: 0.9904 - loss: 0.0262 - val_accuracy: 0.9700 - val_loss: 0.0536
Epoch 4/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 69ms/step - accuracy: 0.9915 - loss: 0.0173 - val_accuracy: 0.9900 - val_loss: 0.0472
Epoch 5/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 65ms/step - accuracy: 0.9913 - loss: 0.0264 - val_accuracy: 0.9900 - val_loss: 0.0594
Epoch 6/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 70ms/step - accuracy: 0.9959 - loss: 0.0186 - val_accuracy: 0.9900 - val_loss: 0.0601
Epoch 7/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9957 - loss: 0.0147 - val_accuracy: 0.9900 - val_loss: 0.0502
Epoch 8/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 63ms/step - accuracy: 0.9969 - loss: 0.0154 - val_accuracy: 0.9900 - val_loss: 0.0494
Epoch 9/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 67ms/step - accuracy: 0.9950 - loss: 0.0146 - val_accuracy: 0.9800 - val_loss: 0.0616
Epoch 10/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 71ms/step - accuracy: 0.9979 - loss: 0.0098 - val_accuracy: 0.9900 - val_loss: 0.0693
Epoch 11/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 67ms/step - accuracy: 0.9975 - loss: 0.0120 - val_accuracy: 0.9900 - val_loss: 0.0575
Epoch 12/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 65ms/step - accuracy: 0.9910 - loss: 0.0211 - val_accuracy: 0.9900 - val_loss: 0.0551
Epoch 13/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9973 - loss: 0.0086 - val_accuracy: 0.9800 - val_loss: 0.0628
Epoch 14/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9960 - loss: 0.0141 - val_accuracy: 0.9800 - val_loss: 0.0744
Epoch 15/15
8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 64ms/step - accuracy: 0.9967 - loss: 0.0077 - val_accuracy: 0.9800 - val_loss: 0.0785


$ ./curlpredict.sh test_img7.png
test_img7.png
{"filename":"test_img7.png","result":"9"}


2024-09-28 05:26:45,591 - main - DEBUG - predict_upload_file starting...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step
INFO:     127.0.0.1:52338 - "POST /predict-image/ HTTP/1.1" 200 OK




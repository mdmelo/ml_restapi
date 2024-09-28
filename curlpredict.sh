#!/bin/bash

set -x

imgfile=${1:-test_img1.png}
echo "$imgfile"

# ./curlpredict.sh test_img2.png | grep result | jq
curl -N -X POST http://127.0.0.1:8000/predict-image/ -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F "file=@$imgfile;type=image/png"


# zipf='/books/MachineLearning/FastAPI/FastAPI-ml-demo/retrain_img.zip'
# file='/books/MachineLearning/FastAPI/FastAPI-ml-demo/test_img2.png'
# url='http://127.0.0.1:8000/predict-image/'
# 
# delim="-----MultipartDelimeter$$$RANDOM$RANDOM$RANDOM"
# nl=$'\r\n'
# mime="$(file -b --mime-type "$file")"
# 
# data() {
#     printf %s "$nl$nl"
#     printf %s "--$delim${nl}Content-Disposition: form-data; name=\"file\"; filename=\"$file\""
#     printf %s "$nl$nl"
#     cat "$file"
#     printf %s "$nl$nl--$delim--$nl"
# }
# 
# response="$(data | curl -v -X POST "$url" -H "content-type: multipart/form-data; boundary=$delim" --data-binary @-)"

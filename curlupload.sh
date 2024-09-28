#!/bin/bash

#set -x

imgzip=${1:-retrain_img.zip}
echo "$imgzip"

# ./curlupload.sh retrain_img.zip | grep message | jq
curl -X POST http://127.0.0.1:8000/upload-images/ -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F "file=@$imgzip;type=application/zip"


# zipf='/books/MachineLearning/FastAPI/FastAPI-ml-demo/retrain_img.zip'
# file='/books/MachineLearning/FastAPI/FastAPI-ml-demo/test_img0.png'
# url='http://127.0.0.1:8000/upload-images/'
# 
# delim="-----MultipartDelimeter$$$RANDOM$RANDOM$RANDOM"
# nl=$'\r\n'
# mime="$(file -b --mime-type "$file")"
# 
# data() {
#     printf %s "$nl$nl"
#     printf %s "--$delim${nl}Content-Disposition: form-data; name=\"file\"; filename=\"$zipf\""
#     printf %s "$nl$nl"
#     cat "$zipf"
#     printf %s "$nl$nl--$delim--$nl"
# }
# 
# response="$(data | curl -v -X POST "$url" -H "content-type: multipart/form-data; boundary=$delim" --data-binary @-)"

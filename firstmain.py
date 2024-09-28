from fastapi import FastAPI, UploadFile, BackgroundTasks
import logging
import sys


# run the app via 'fastapi run firstmain.py'
# see https://fastapi.tiangolo.com/deployment/manually/

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.debug('1 this is a debug message')
app = FastAPI(debug=True)
logger.debug('2 this is a debug message')


@app.get("/")
async def root():
    logger.debug('3 this is a debug message')    
    return {"message": "Hello WTF!"}



@app.get("/hello/{name}")
async def say_hello(name: str):
    logger.debug('4 this is a debug message')    
    return {"message": f"Hello {name}"}


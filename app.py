from transformers import pipeline
import gradio as gr
from pytube import YouTube

pipe = pipeline(model="howlbz/whisper-small-hi")  # change to "your-username/the-name-you-picked"

def transcribe(audio,url):
    if url:
      youtubeObject = YouTube(url).streams.first().download()
      audio = youtubeObject
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=[
            gr.Audio(source="microphone", type="filepath"),
            gr.inputs.Textbox(label="give me an url",default ="https://www.youtube.com/watch?v=YzGsIavAo_E")
            ],
    outputs="text",
    title="Whisper Small Chinese",
    description="Realtime demo for chinese speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()



# import gradio as gr
# import numpy as np
# from PIL import Image
# import requests
#
# import hopsworks
# import joblib
#
# project = hopsworks.login()
# fs = project.get_feature_store()
#
# #HwJaWmtvaCzFra3g.89QYueFGuScRnJkiepzG2tiWtKSrqNHCCJrnVie9fwhIMeJxRUpAGAT7mF36MDMv
# mr = project.get_model_registry()
# model = mr.get_model("iris_modal", version=1)
# model_dir = model.download()
# model = joblib.load(model_dir + "/iris_model.pkl")
#
#
# def iris(sepal_length, sepal_width, petal_length, petal_width):
#     input_list = []
#     input_list.append(sepal_length)
#     input_list.append(sepal_width)
#     input_list.append(petal_length)
#     input_list.append(petal_width)
#     # 'res' is a list of predictions returned as the label.
#     res = model.predict(np.asarray(input_list).reshape(1, -1))
#     # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
#     # the first element.
#     flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
#     img = Image.open(requests.get(flower_url, stream=True).raw)
#     return img
#
# demo = gr.Interface(
#     fn=iris,
#     title="Iris Flower Predictive Analytics",
#     description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
#     allow_flagging="never",
#     inputs=[
#         gr.inputs.Number(default=1.0, label="sepal length (cm)"),
#         gr.inputs.Number(default=1.0, label="sepal width (cm)"),
#         gr.inputs.Number(default=1.0, label="petal length (cm)"),
#         gr.inputs.Number(default=1.0, label="petal width (cm)"),
#         ],
#     outputs=gr.Image(type="pil"))
#
# demo.launch(share = True)
#

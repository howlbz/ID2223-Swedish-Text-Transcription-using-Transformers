# Lab2 Swedish Text Transcription using Transformers

The goal of this assignment is to fine-tune a pre-trained transformer model, Whisper, to transcribe Swedish language audio (or audio your mother tongue) to text. You should provide a user interface to allow users to use your model, providing some useful or entertaining service. You should also explain how to improve the performance of your model with either model-centric improvements or data centric improvements.

## Hugging Face Space

https://huggingface.co/spaces/howlbz/lab2

## Ways to improve model performance
### Model-centric approach: 

We can increase the learning rate so that the optimization will be faster. We can also increase the per_device_train_batch_size and decrease gradient_accumulation_steps at the same time since they are highly connected. When more data are used in each step, the gradient changes slower. Besides, changing the size of the Whisper checkpoints from small to medium or large is also a possible way to improve the performance as the model is larger and more complexed. 

### Data-centric approach:

The following webpage contains Chinese dataset from other source that may improve the performance:

https://www.twine.net/blog/mandarin-chinese-langauge-datasets/


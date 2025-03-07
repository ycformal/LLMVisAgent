Question: What number is on the front of the train?

Reference Answer: 165026

Image path: ./sampled_GQA/525361.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='train')
IMAGE0=CROP_FRONT(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What number is on the train?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="What number is on the front of the train?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What number is on the front of the train?')=<b><span style='color: green;'>150206</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>150206</span></b></div><hr>

Answer: 165026


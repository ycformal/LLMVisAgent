Question: What color is her hair?

Reference Answer: brown

Image path: ./sampled_GQA/394199.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='woman')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is her hair?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="What color is her hair?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What color is her hair?')=<b><span style='color: green;'>red</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>red</span></b></div><hr>

Answer: brown


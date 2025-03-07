Question: Who is standing?

Reference Answer: snowboarder

Image path: ./sampled_GQA/n352479.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='standing')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Who is standing?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Who is standing?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Who is standing?')=<b><span style='color: green;'>woman</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>woman</span></b></div><hr>

Answer: woman


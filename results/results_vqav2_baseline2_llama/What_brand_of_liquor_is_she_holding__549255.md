Question: What brand of liquor is she holding?

Reference Answer: bacardi

Image path: ./sampled_GQA/549255.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='woman')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='liquor')
IMAGE1=CROP(image=IMAGE0,box=BOX1)
ANSWER0=VQA(image=IMAGE1,question='What brand of liquor is she holding?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="What brand of liquor is she holding?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What brand of liquor is she holding?')=<b><span style='color: green;'>blackbeard</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>blackbeard</span></b></div><hr>

Answer: Bacardi


Question: Do both the people have the same gender?

Reference Answer: no

Image path: ./sampled_GQA/n541688.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='people')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='male')
IMAGE1=CROP(image=IMAGE,box=BOX1)
BOX2=LOC(image=IMAGE0,object='female')
IMAGE2=CROP(image=IMAGE,box=BOX2)
ANSWER0=COUNT(box=BOX1)
ANSWER1=COUNT(box=BOX2)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Do both the people have the same gender?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Do both the people have the same gender?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: No


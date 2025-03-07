Question: Do they both have sunglasses?

Reference Answer: no

Image path: ./sampled_GQA/578492.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='person')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='sunglasses')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Do they both have sunglasses?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Do they both have sunglasses?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


Question: Has anything melted on the plates?

Reference Answer: no

Image path: ./sampled_GQA/99810.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='plate')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='melted')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Has anything melted on the plates?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Has anything melted on the plates?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


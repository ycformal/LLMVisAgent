Question: Do you see any green skateboards or bicycles?

Reference Answer: no

Image path: ./sampled_GQA/n259949.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='skateboard')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='green')
ANSWER0=COUNT(box=BOX1)
BOX2=LOC(image=IMAGE,object='bicycle')
IMAGE1=CROP(image=IMAGE,box=BOX2)
BOX3=LOC(image=IMAGE1,object='green')
ANSWER1=COUNT(box=BOX3)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} > 0 or {ANSWER1} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Do you see any green skateboards or bicycles?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Do you see any green skateboards or bicycles?')=<b><span style='color: green;'>skateboards</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>skateboards</span></b></div><hr>

Answer: skateboards


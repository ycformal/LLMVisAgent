Question: Is there any bacon in this picture that is long?

Reference Answer: yes

Image path: ./sampled_GQA/n170941.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='bacon')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='long')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is there any bacon in this picture that is long?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is there any bacon in this picture that is long?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: Yes


Question: Is that a White Bear?

Reference Answer: no

Image path: ./sampled_GQA/372652.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='bear')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the bear?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'white' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is that a White Bear?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is that a White Bear?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: no


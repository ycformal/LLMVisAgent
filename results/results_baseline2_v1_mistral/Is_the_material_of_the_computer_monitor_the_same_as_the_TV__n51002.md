Question: Is the material of the computer monitor the same as the TV?

Reference Answer: yes

Image path: ./sampled_GQA/n51002.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='computer monitor')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object='tv')
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question='What material is the computer monitor made of?')
ANSWER1=VQA(image=IMAGE1,question='What material is the tv made of?')
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is the material of the computer monitor the same as the TV?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the material of the computer monitor the same as the TV?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


Question: Is this bathroom being remodeled?

Reference Answer: no

Image path: ./sampled_GQA/62491.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='bathroom')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='remodel')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is this bathroom being remodeled?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is this bathroom being remodeled?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


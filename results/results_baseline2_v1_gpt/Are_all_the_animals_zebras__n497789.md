Question: Are all the animals zebras?

Reference Answer: no

Image path: ./sampled_GQA/n497789.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='zebra')
ANSWER0=COUNT(box=BOX0)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == {TOTAL_OBJECTS} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Are all the animals zebras?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Are all the animals zebras?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: no


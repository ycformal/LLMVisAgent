Question: Is that floor blue?

Reference Answer: no

Image path: ./sampled_GQA/n534106.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What color is the floor?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'blue' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What color is the floor?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'blue' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What color is the floor?')=<b><span style='color: green;'>brown</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="'yes' if ANSWER0 == 'blue' else 'no'")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="'yes' if 'brown' == 'blue' else 'no'")=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


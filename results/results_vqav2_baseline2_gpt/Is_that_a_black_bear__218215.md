Question: Is that a black bear?

Reference Answer: no

Image path: ./sampled_GQA/218215.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What animal is in the image?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'black bear' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What animal is in the image?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'black bear' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What animal is in the image?')=<b><span style='color: green;'>polar bear</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="'yes' if ANSWER0 == 'black bear' else 'no'")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="'yes' if 'polar bear' == 'black bear' else 'no'")=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


Question: What sport is this man playing?

Reference Answer: frisbee

Image path: ./sampled_GQA/297970.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What sport is this man playing?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What sport is this man playing?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What sport is this man playing?')=<b><span style='color: green;'>frisbee</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>frisbee</span></b></div><hr>

Answer: frisbee


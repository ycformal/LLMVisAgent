Question: Has it been raining?

Reference Answer: no

Image path: ./sampled_GQA/299083.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Has it been raining?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Has it been raining?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Has it been raining?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


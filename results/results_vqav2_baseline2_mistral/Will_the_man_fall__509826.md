Question: Will the man fall?

Reference Answer: no

Image path: ./sampled_GQA/509826.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Will the man fall?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Will the man fall?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Will the man fall?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


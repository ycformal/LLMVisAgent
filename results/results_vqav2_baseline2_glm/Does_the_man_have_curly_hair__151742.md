Question: Does the man have curly hair?

Reference Answer: no

Image path: ./sampled_GQA/151742.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Does the man have curly hair?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Does the man have curly hair?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Does the man have curly hair?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


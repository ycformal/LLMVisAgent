Question: How many people are in the picture?

Reference Answer: 2

Image path: ./sampled_GQA/356094.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='How many people are in the picture?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='How many people are in the picture?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many people are in the picture?')=<b><span style='color: green;'>2</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>2</span></b></div><hr>

Answer: 2


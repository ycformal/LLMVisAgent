Question: How thick is the knife?

Reference Answer: thin

Image path: ./sampled_GQA/n100991.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='How thick is the knife?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='How thick is the knife?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How thick is the knife?')=<b><span style='color: green;'>not very</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>not very</span></b></div><hr>

Answer: Not very


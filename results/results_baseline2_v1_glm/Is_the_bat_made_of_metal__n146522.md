Question: Is the bat made of metal?

Reference Answer: no

Image path: ./sampled_GQA/n146522.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Is the bat made of metal?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Is the bat made of metal?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the bat made of metal?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: Yes


Question: Is this man playing a game?

Reference Answer: yes

Image path: ./sampled_GQA/392981.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Is this man playing a game?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Is this man playing a game?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is this man playing a game?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: Yes


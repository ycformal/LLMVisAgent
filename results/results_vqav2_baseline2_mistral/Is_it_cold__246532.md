Question: Is it cold?

Reference Answer: yes

Image path: ./sampled_GQA/246532.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Is it cold?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Is it cold?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is it cold?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: yes


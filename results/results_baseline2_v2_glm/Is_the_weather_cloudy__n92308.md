Question: Is the weather cloudy?

Reference Answer: yes

Image path: ./sampled_GQA/n92308.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Is the weather cloudy?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Is the weather cloudy?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the weather cloudy?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: No


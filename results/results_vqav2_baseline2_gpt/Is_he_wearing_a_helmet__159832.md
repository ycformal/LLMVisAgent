Question: Is he wearing a helmet?

Reference Answer: no

Image path: ./sampled_GQA/159832.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Is he wearing a helmet?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Is he wearing a helmet?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is he wearing a helmet?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: no


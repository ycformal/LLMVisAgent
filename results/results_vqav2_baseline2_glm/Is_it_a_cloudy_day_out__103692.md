Question: Is it a cloudy day out?

Reference Answer: yes

Image path: ./sampled_GQA/103692.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Is it a cloudy day out?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Is it a cloudy day out?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is it a cloudy day out?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: No


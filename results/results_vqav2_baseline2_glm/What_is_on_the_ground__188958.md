Question: What is on the ground?

Reference Answer: snow

Image path: ./sampled_GQA/188958.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What is on the ground?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What is on the ground?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What is on the ground?')=<b><span style='color: green;'>snow</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>snow</span></b></div><hr>

Answer: snow


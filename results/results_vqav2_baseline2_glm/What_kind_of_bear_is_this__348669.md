Question: What kind of bear is this?

Reference Answer: grizzly

Image path: ./sampled_GQA/348669.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What kind of bear is this?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What kind of bear is this?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What kind of bear is this?')=<b><span style='color: green;'>brown</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>brown</span></b></div><hr>

Answer: brown


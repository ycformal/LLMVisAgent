Question: What color are the shoes?

Reference Answer: green

Image path: ./sampled_GQA/240681.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What color are the shoes?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What color are the shoes?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What color are the shoes?')=<b><span style='color: green;'>green</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>green</span></b></div><hr>

Answer: green


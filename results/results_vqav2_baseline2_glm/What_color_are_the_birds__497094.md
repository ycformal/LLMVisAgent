Question: What color are the birds?

Reference Answer: blue

Image path: ./sampled_GQA/497094.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What color are the birds?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What color are the birds?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What color are the birds?')=<b><span style='color: green;'>blue</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>blue</span></b></div><hr>

Answer: blue


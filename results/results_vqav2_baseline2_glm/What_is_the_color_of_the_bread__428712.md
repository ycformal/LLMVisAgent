Question: What is the color of the bread?

Reference Answer: white

Image path: ./sampled_GQA/428712.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What is the color of the bread?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What is the color of the bread?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What is the color of the bread?')=<b><span style='color: green;'>white</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>white</span></b></div><hr>

Answer: white


Question: Where is the television?

Reference Answer: on wall

Image path: ./sampled_GQA/560312.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Where is the television?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Where is the television?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Where is the television?')=<b><span style='color: green;'>on wall</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>on wall</span></b></div><hr>

Answer: above the window on the wall


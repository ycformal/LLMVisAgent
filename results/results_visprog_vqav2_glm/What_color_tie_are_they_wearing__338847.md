Question: What color tie are they wearing?

Reference Answer: red

Image path: ./sampled_GQA/338847.jpg

Program:

```
ANSWER0=VQA(image=IMAGE,question='What color tie are they wearing?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What color tie are they wearing?')=<b><span style='color: green;'>red</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>red</span></b></div><hr>

Answer: red


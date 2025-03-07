Question: How many legs does the giraffe have?

Reference Answer: 4

Image path: ./sampled_GQA/398580.jpg

Program:

```
ANSWER0=VQA(image=IMAGE,question='How many legs does the giraffe have?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many legs does the giraffe have?')=<b><span style='color: green;'>4</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>4</span></b></div><hr>

Answer: 4


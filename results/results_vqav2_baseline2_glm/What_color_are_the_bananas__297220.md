Question: What color are the bananas?

Reference Answer: yellow

Image path: ./sampled_GQA/297220.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What color are the bananas?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What color are the bananas?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What color are the bananas?')=<b><span style='color: green;'>yellow</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yellow</span></b></div><hr>

Answer: yellow


Question: What type of animal is in the picture?

Reference Answer: elephant

Image path: ./sampled_GQA/301061.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What type of animal is in the picture?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What type of animal is in the picture?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What type of animal is in the picture?')=<b><span style='color: green;'>elephant</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>elephant</span></b></div><hr>

Answer: elephant


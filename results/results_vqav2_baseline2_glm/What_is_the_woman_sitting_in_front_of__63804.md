Question: What is the woman sitting in front of?

Reference Answer: laptop

Image path: ./sampled_GQA/63804.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What is the woman sitting in front of?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What is the woman sitting in front of?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What is the woman sitting in front of?')=<b><span style='color: green;'>laptop</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>laptop</span></b></div><hr>

Answer: laptop


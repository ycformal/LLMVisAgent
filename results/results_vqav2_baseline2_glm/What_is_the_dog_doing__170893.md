Question: What is the dog doing?

Reference Answer: chewing toilet paper

Image path: ./sampled_GQA/170893.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What is the dog doing?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What is the dog doing?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What is the dog doing?')=<b><span style='color: green;'>looking in toilet</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>looking in toilet</span></b></div><hr>

Answer: looking in toilet


Question: Who is the person that generally takes care of this type of animal?

Reference Answer: shepherd

Image path: ./sampled_GQA/434060.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Who is the person that generally takes care of this type of animal?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Who is the person that generally takes care of this type of animal?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Who is the person that generally takes care of this type of animal?')=<b><span style='color: green;'>farmer</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>farmer</span></b></div><hr>

Answer: farmer


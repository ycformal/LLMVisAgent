Question: Which kind of material was used to make this fence?

Reference Answer: metal

Image path: ./sampled_GQA/n222297.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Which kind of material was used to make this fence?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Which kind of material was used to make this fence?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Which kind of material was used to make this fence?')=<b><span style='color: green;'>metal</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>metal</span></b></div><hr>

Answer: metal


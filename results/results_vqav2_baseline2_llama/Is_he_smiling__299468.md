Question: Is he smiling?

Reference Answer: no

Image path: ./sampled_GQA/299468.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='man')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Is he smiling?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is he smiling?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is he smiling?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: Yes.


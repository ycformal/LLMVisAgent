Question: Does the skier that is skiing appear to be male?

Reference Answer: yes

Image path: ./sampled_GQA/n143672.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Does the skier that is skiing appear to be male?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Does the skier that is skiing appear to be male?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Does the skier that is skiing appear to be male?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: yes


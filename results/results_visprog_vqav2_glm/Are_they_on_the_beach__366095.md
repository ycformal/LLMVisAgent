Question: Are they on the beach?

Reference Answer: yes

Image path: ./sampled_GQA/366095.jpg

Program:

```
ANSWER0=VQA(image=IMAGE,question='Are they on the beach?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Are they on the beach?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: yes


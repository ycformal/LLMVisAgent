Question: How is the weather?

Reference Answer: clear

Image path: ./sampled_GQA/550365.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='How is the weather?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='How is the weather?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How is the weather?')=<b><span style='color: green;'>cold</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>cold</span></b></div><hr>

Answer: clear and sunny


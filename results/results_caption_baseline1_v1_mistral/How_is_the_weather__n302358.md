Question: How is the weather?

Reference Answer: clear

Image path: ./sampled_GQA/n302358.jpg

Program:

```
ANSWER0=VQA(image=IMAGE,question='How is the weather?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How is the weather?')=<b><span style='color: green;'>cloudy</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>cloudy</span></b></div><hr>

Answer: Sunny and calm.


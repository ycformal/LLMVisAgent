Question: The right image shows an afghan hound with a bushy curling tail posed with its prey in front of it.

Reference Answer: True

Left image URL: https://s-media-cache-ak0.pinimg.com/originals/94/f6/5a/94f65a39c83de65e87ce7ebaf3192b94.jpg

Right image URL: https://s-media-cache-ak0.pinimg.com/originals/fb/11/17/fb111780675e433d231c5e0d799b0746.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='Is the image an afghan hound with a bushy curling tail posed with its prey in front of it?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='Is the image an afghan hound with a bushy curling tail posed with its prey in front of it?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the image an afghan hound with a bushy curling tail posed with its prey in front of it?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="True")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


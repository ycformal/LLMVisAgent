Question: In the image to the right, the wine bottle is NOT standing evenly on any type of surface that we can see.

Reference Answer: False

Left image URL: https://s-media-cache-ak0.pinimg.com/736x/97/f8/ae/97f8aec564d457ba330e5bf75bcb9664--nice-body-chilli.jpg

Right image URL: http://www.kristalamb.com/wp-content/uploads/2016/09/Fall1.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='Is the wine bottle standing evenly on any type of surface that we can see?')
ANSWER1=EVAL(expr='not {ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='Is the wine bottle standing evenly on any type of surface that we can see?')
ANSWER1=EVAL(expr='not {ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the wine bottle standing evenly on any type of surface that we can see?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="not ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="not True")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


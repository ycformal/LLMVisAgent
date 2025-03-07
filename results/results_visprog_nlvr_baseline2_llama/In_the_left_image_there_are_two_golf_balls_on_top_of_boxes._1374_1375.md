Question: In the left image there are two golf balls on top of boxes.

Reference Answer: True

Left image URL: http://www.mygolfway.com/wp-content/uploads/2015/02/ProV1_ProV1X_group_RGB.jpg

Right image URL: https://http2.mlstatic.com/caja-de-06-pelotas-titleist-pro-v1-D_NQ_NP_550915-MLV25335888966_022017-F.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many golf balls are on top of boxes?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many golf balls are on top of boxes?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many golf balls are on top of boxes?')=<b><span style='color: green;'>2</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 2")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="2 == 2")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


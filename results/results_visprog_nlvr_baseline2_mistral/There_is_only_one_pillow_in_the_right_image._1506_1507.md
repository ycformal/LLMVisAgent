Question: There is only one pillow in the right image.

Reference Answer: False

Left image URL: https://s-media-cache-ak0.pinimg.com/736x/53/a3/1f/53a31f5d6804a88c68a838901190ecc5.jpg

Right image URL: http://www.definingelegance.com/media/Kevin_OBrien/Mohair/PI_BoxPillows-Stacked.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='How many pillows are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='How many pillows are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many pillows are in the image?')=<b><span style='color: green;'>5</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 1")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="5 == 1")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


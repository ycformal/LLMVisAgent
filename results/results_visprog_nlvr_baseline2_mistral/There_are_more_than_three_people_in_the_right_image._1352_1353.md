Question: There are more than three people in the right image.

Reference Answer: False

Left image URL: https://s3-media1.fl.yelpcdn.com/bphoto/p8bBJyOW66EU5qqWePTqIQ/ls.jpg

Right image URL: https://buffetoblog.files.wordpress.com/2009/01/guys-eating-huge-burgers.jpg?w=652

Original program:

```
ANSWER0=VQA(image=RIGHT,question='How many people are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} > 3')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='How many people are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} > 3')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many people are in the image?')=<b><span style='color: green;'>3</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 > 3")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="3 > 3")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


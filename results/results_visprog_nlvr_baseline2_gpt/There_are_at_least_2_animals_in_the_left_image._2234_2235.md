Question: There are at least 2 animals in the left image.

Reference Answer: True

Left image URL: https://leesburgvetblog.files.wordpress.com/2014/02/guinea-pigs_vanetta-and-sully.jpg?w=640&h=427&crop=1

Right image URL: https://farm4.staticflickr.com/3257/2864103965_051cc6d855.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many animals are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} >= 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many animals are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} >= 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many animals are in the image?')=<b><span style='color: green;'>2</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 >= 2")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="2 >= 2")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


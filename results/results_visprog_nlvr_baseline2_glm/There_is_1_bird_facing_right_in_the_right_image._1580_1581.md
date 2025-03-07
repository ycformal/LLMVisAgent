Question: There is 1 bird facing right in the right image.

Reference Answer: True

Left image URL: https://stevetabone.files.wordpress.com/2012/01/white-pelican.jpg

Right image URL: https://naturetime.files.wordpress.com/2014/01/white-pelican-swimming.jpg?w=890

Original program:

```
ANSWER0=VQA(image=RIGHT,question='How many birds are facing right?')
ANSWER1=EVAL(expr='{ANSWER0} == 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='How many birds are facing right?')
ANSWER1=EVAL(expr='{ANSWER0} == 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many birds are facing right?')=<b><span style='color: green;'>1</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 1")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="1 == 1")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


Question: The left image contains exactly two dogs.

Reference Answer: False

Left image URL: https://c1.staticflickr.com/1/187/377244827_963a59b12c_b.jpg

Right image URL: http://www.foodsafetynews.com/files/2014/05/Dog-on-couch.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many dogs are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many dogs are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many dogs are in the image?')=<b><span style='color: green;'>1</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 2")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="1 == 2")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


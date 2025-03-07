Question: At least one sail boat is parked on the left side of the dock.

Reference Answer: False

Left image URL: http://www.denmanmarine.com.au/image/data/Sail/Caledonia%20Yawl/EowynLaunch050.jpg

Right image URL: https://i.pinimg.com/originals/c9/bc/de/c9bcde3ab9b1964d7b20e45a71f9aaf3.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many sail boats are parked on the left side of the dock?')
ANSWER1=EVAL(expr='{ANSWER0} >= 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many sail boats are parked on the left side of the dock?')
ANSWER1=EVAL(expr='{ANSWER0} >= 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many sail boats are parked on the left side of the dock?')=<b><span style='color: green;'>1</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 >= 1")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="1 >= 1")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


Question: There are multiple colored safety pins threaded onto one red safety pin.

Reference Answer: True

Left image URL: https://ae01.alicdn.com/kf/HTB1enTMNVXXXXcMXpXXq6xXFXXX2/25pcs-pack-11-16-Plastic-MIX-Colorful-Safety-Pins-For-Label-Tags-Fastener-Charms-Baby-Shower.jpg

Right image URL: http://www.sinopos.com//upload/20150924/20150924171039_86183.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='Are there multiple colored safety pins threaded onto one red safety pin?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='Are there multiple colored safety pins threaded onto one red safety pin?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Are there multiple colored safety pins threaded onto one red safety pin?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="False")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


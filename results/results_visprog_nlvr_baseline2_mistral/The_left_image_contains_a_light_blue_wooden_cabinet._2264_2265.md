Question: The left image contains a light blue wooden cabinet.

Reference Answer: False

Left image URL: https://i.pinimg.com/736x/3e/9e/6e/3e9e6ef6ceda33c63093c4b02eed6816--painted-china-cabinets-painted-hutch.jpg

Right image URL: https://s-media-cache-ak0.pinimg.com/originals/02/cf/34/02cf34311d3b9627413f4c99cce97e01.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='Does the image contain a light blue wooden cabinet?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='Does the image contain a light blue wooden cabinet?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Does the image contain a light blue wooden cabinet?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="False")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


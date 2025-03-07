Question: There is exactly one chimpanzee in the left image.

Reference Answer: True

Left image URL: http://www.chinadaily.com.cn/world/2006-05/19/xin_110503191100177163659.jpg

Right image URL: https://i.ytimg.com/vi/cy9xc2hz_y4/maxresdefault.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many chimpanzees are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many chimpanzees are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many chimpanzees are in the image?')=<b><span style='color: green;'>1</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 1")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="1 == 1")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


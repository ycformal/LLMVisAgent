Question: The left image contains exactly two pandas.

Reference Answer: True

Left image URL: https://polydidacticism.files.wordpress.com/2015/03/pan-porn.jpg

Right image URL: https://s-media-cache-ak0.pinimg.com/originals/60/e1/ce/60e1ce085d86fb4dfc28c3bee6840de8.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many pandas are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many pandas are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many pandas are in the image?')=<b><span style='color: green;'>2</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 2")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="2 == 2")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


Question: The name of the lotion in the right image is two words.

Reference Answer: False

Left image URL: https://i.ebayimg.com/images/g/DLoAAOSw0yxZt0QW/s-l300.jpg

Right image URL: https://2.bp.blogspot.com/-dMZD4fpTFVk/WcaEA9AKDtI/AAAAAAAAGoE/RiQV7AfBk3QXSa6RK2dHaBSacEjEJT2jQCLcBGAs/s1600/lush-sleepy-lotion.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='How many words are in the name of the lotion?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='How many words are in the name of the lotion?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many words are in the name of the lotion?')=<b><span style='color: green;'>4</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 2")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="4 == 2")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


Question: The image on the right contains no more than one dog.

Reference Answer: True

Left image URL: https://s-media-cache-ak0.pinimg.com/originals/d3/c3/26/d3c3268d47c4551aeea94c4b3f177813.jpg

Right image URL: https://lh4.ggpht.com/PCALybLkx7oxpNAa4B2ws24exaAoWQDD4dfUpT_g3zD8EL0sy4bsnYXXkEbhHJolZA=h900

Original program:

```
ANSWER0=VQA(image=RIGHT,question='How many dogs are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} <= 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='How many dogs are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} <= 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many dogs are in the image?')=<b><span style='color: green;'>1</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 <= 1")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="1 <= 1")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


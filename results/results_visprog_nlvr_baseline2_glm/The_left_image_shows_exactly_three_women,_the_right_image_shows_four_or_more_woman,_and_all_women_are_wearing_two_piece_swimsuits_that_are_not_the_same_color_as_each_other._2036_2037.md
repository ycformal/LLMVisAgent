Question: The left image shows exactly three women, the right image shows four or more woman, and all women are wearing two piece swimsuits that are not the same color as each other.

Reference Answer: False

Left image URL: https://s-media-cache-ak0.pinimg.com/originals/ee/77/d4/ee77d4df6d0e58c0949ed48ff95d1ec8.jpg

Right image URL: https://i.pinimg.com/originals/99/23/ed/9923ed812d85cb366a195ba2b56d3f2d.jpg

Original program:

```
Statement: The left image shows exactly three women, the right image shows four or more woman, and all women are wearing two piece swimsuits that are not the same color as each other.
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is 'The left image shows exactly three women, the right image shows four or more woman, and all women are wearing two piece swimsuits that are not the same color as each other.' true or false?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is 'The left image shows exactly three women, the right image shows four or more woman, and all women are wearing two piece swimsuits that are not the same color as each other.' true or false?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: False


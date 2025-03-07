Question: The truck in the image on the right is on a wet surface.

Reference Answer: True

Left image URL: http://wtop.com/wp-content/uploads/2015/11/IMG_5478.jpg

Right image URL: https://bdn-data.s3.amazonaws.com/uploads/2010/09/1283601116_6f9a-1024x664.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='Is the truck on a wet surface?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='Is the truck on a wet surface?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the truck on a wet surface?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="True")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


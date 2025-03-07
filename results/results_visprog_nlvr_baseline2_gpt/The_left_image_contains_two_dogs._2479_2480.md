Question: The left image contains two dogs.

Reference Answer: True

Left image URL: http://s3.amazonaws.com/medias.photodeck.com/20945f3d-ef98-4b7b-b029-9f99290b73ea/dwss-160_medium.jpg

Right image URL: https://i.pinimg.com/736x/90/76/85/90768587c680b66eaaec2d541fe19052--dog-videos-afghan-hound.jpg

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

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many dogs are in the image?')=<b><span style='color: green;'>2</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 2")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="2 == 2")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


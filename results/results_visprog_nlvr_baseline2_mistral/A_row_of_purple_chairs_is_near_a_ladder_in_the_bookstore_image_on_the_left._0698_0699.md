Question: A row of purple chairs is near a ladder in the bookstore image on the left.

Reference Answer: True

Left image URL: https://media-cdn.tripadvisor.com/media/photo-s/0f/29/27/fa/full-circle-bookstore.jpg

Right image URL: https://s3-media1.fl.yelpcdn.com/bphoto/8cozfnwh3ob-goNrAi9cMQ/348s.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='Is there a row of purple chairs near a ladder?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='Is there a row of purple chairs near a ladder?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is there a row of purple chairs near a ladder?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="True")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


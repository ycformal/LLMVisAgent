Question: The bookshelf in the image on the right is angled into a triangular shape.

Reference Answer: False

Left image URL: http://hgtvhome.sndimg.com/content/dam/images/hgtv/fullset/2015/4/20/0/CI-Tamara-H-Design_Bunkbed-Staircase-Bookcase.jpg.rend.hgtvcom.966.1208.suffix/1429568578522.jpeg

Right image URL: https://cdn.makespace.com/blog/wp-content/uploads/2016/02/09131121/bookshelf-room-divider-book-storage-hack.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='Is the bookshelf angled into a triangular shape?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='Is the bookshelf angled into a triangular shape?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the bookshelf angled into a triangular shape?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="True")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


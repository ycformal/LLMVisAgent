Question: There is at least one utensil laying on the plate in the image on the right.

Reference Answer: True

Left image URL: https://media-cdn.tripadvisor.com/media/photo-s/02/30/13/22/popcorn-chicken-macaroni.jpg

Right image URL: https://media-cdn.tripadvisor.com/media/photo-s/07/9d/67/09/fried-chicken-mac-n-cheese.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='Is there at least one utensil laying on the plate?')
FINAL_ANSWER=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='Is there at least one utensil laying on the plate?')
FINAL_ANSWER=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is there at least one utensil laying on the plate?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>yes</span></b></div><hr>

Answer: True


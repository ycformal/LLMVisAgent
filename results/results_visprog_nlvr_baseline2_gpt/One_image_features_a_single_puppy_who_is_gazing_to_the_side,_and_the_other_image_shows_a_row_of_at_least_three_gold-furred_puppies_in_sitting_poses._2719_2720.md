Question: One image features a single puppy who is gazing to the side, and the other image shows a row of at least three gold-furred puppies in sitting poses.

Reference Answer: True

Left image URL: https://balsambranchkenneldotnet.files.wordpress.com/2017/02/fox-red-lab-puppies-balsam-branch-kennel-trb-5wks-females.jpg?w=620

Right image URL: https://s-media-cache-ak0.pinimg.com/736x/9b/1c/fd/9b1cfd4f36bdd71f278d52f28b8c0526--red-labrador-labrador-retriever.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many puppies are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many puppies are in the image?')
ANSWER2=VQA(image=LEFT,question='Is the puppy gazing to the side?')
ANSWER3=VQA(image=RIGHT,question='Is the puppy gazing to the side?')
ANSWER4=VQA(image=LEFT,question='Are there at least three gold-furred puppies in sitting poses?')
ANSWER5=VQA(image=RIGHT,question='Are there at least three gold-furred puppies in sitting poses?')
ANSWER6=EVAL(expr='{ANSWER0} == 1 and {ANSWER2}')
ANSWER7=EVAL(expr='{ANSWER1} >= 3 and {ANSWER4}')
ANSWER8=EVAL(expr='{ANSWER6} xor {ANSWER7}')
FINAL_ANSWER=RESULT(var=ANSWER8)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many puppies are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many puppies are in the image?')
ANSWER2=VQA(image=LEFT,question='Is the puppy gazing to the side?')
ANSWER3=VQA(image=RIGHT,question='Is the puppy gazing to the side?')
ANSWER4=VQA(image=LEFT,question='Are there at least three gold-furred puppies in sitting poses?')
ANSWER5=VQA(image=RIGHT,question='Are there at least three gold-furred puppies in sitting poses?')
ANSWER6=EVAL(expr='{ANSWER0} == 1 and {ANSWER2}')
ANSWER7=EVAL(expr='{ANSWER1} >= 3 and {ANSWER4}')
ANSWER8=EVAL(expr='{ANSWER6} xor {ANSWER7}')
FINAL_ANSWER=RESULT(var=ANSWER8)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many puppies are in the image?')=<b><span style='color: green;'>3</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many puppies are in the image?')=<b><span style='color: green;'>1</span></b></div><hr><div><b><span style='color: blue;'>ANSWER2</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the puppy gazing to the side?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER3</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the puppy gazing to the side?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER4</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Are there at least three gold-furred puppies in sitting poses?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER5</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Are there at least three gold-furred puppies in sitting poses?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: blue;'>ANSWER6</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 1 and ANSWER2")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="3 == 1 and True")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: blue;'>ANSWER7</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER1 >= 3 and ANSWER4")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="1 >= 3 and True")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: blue;'>ANSWER8</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER6 xor ANSWER7")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="False != False")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER8</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


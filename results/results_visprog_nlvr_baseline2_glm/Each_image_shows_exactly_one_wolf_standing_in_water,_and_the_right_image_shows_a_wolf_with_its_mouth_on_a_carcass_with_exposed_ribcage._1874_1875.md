Question: Each image shows exactly one wolf standing in water, and the right image shows a wolf with its mouth on a carcass with exposed ribcage.

Reference Answer: False

Left image URL: https://i.pinimg.com/736x/b9/ae/f9/b9aef9217ba29a3304f7f3a77cea0232--wolf-jewelry-baby-wolves.jpg

Right image URL: https://i.pinimg.com/736x/16/7f/f8/167ff844823c65c5e13ed64f7fe3876f--eye-wolves.jpg

Original program:

```
Statement: An image shows one bare hand with the thumb on the right holding up a belly-first, head-up crab, with water in the background.
Program:
ANSWER0=VQA(image=LEFT,question='Does the image shows one bare hand with the thumb on the right holding a crab?')
ANSWER1=VQA(image=RIGHT,question='Does the image shows one bare hand with the thumb on the right holding a crab?')
ANSWER2=VQA(image=LEFT,question='Is the crab belly-first and head-ups?')
ANSWER3=VQA(image=RIGHT,question='Is the crab belly-first and head-ups?')
ANSWER4=VQA(image=LEFT,question='Is there water in the background?')
ANSWER5=VQA(image=RIGHT,question='Is there water in the background?')
ANSWER6=EVAL(expr='{ANSWER0} and {ANSWER2} and {ANSWER4}')
ANSWER7=EVAL(expr='{ANSWER1} and {ANSWER3} and {ANSWER5}')
ANSWER8=EVAL(expr='{ANSWER6} xor {ANSWER7}')
FINAL_ANSWER=RESULT(var=ANSWER8)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is 'Each image shows exactly one wolf standing in water, and the right image shows a wolf with its mouth on a carcass with exposed ribcage.' true or false?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is 'Each image shows exactly one wolf standing in water, and the right image shows a wolf with its mouth on a carcass with exposed ribcage.' true or false?')=<b><span style='color: green;'>false</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>false</span></b></div><hr>

Answer: false


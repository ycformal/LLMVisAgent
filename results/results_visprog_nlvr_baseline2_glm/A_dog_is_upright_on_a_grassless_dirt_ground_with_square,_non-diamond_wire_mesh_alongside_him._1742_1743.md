Question: A dog is upright on a grassless dirt ground with square, non-diamond wire mesh alongside him.

Reference Answer: True

Left image URL: https://s-media-cache-ak0.pinimg.com/originals/11/71/0f/11710fbb10d1589a9d896a3a16ba7e13.jpg

Right image URL: https://s-media-cache-ak0.pinimg.com/originals/4c/31/f7/4c31f79beae881bb309842ab27c17b6e.jpg

Original program:

```
Statement: A dog is upright on a grassless dirt ground with square, non-diamond wire mesh alongside him.
Program:
ANSWER0=VQA(image=LEFT,question='Is the dog upright on a grassless dirt ground?')
ANSWER1=VQA(image=RIGHT,question='Is the dog upright on a grassless dirt ground?')
ANSWER2=VQA(image=LEFT,question='Is there square, non-diamond wire mesh alongside him?')
ANSWER3=VQA(image=RIGHT,question='Is there square, non-diamond wire mesh alongside him?')
ANSWER4=EVAL(expr='{ANSWER0} and {ANSWER2}')
ANSWER5=EVAL(expr='{ANSWER1} and {ANSWER3}')
ANSWER6=EVAL(expr='{ANSWER4} xor {ANSWER5}')
FINAL_ANSWER=RESULT(var=ANSWER6)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is 'A dog is upright on a grassless dirt ground with square, non-diamond wire mesh alongside him.' true or false?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is 'A dog is upright on a grassless dirt ground with square, non-diamond wire mesh alongside him.' true or false?')=<b><span style='color: green;'>true</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>true</span></b></div><hr>

Answer: true


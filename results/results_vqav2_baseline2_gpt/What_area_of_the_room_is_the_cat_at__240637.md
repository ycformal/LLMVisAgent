Question: What area of the room is the cat at?

Reference Answer: bed

Image path: ./sampled_GQA/240637.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='cat')
ANSWER0=AREA(box=BOX0)
ANSWER1=EVAL(expr="'top left' if {ANSWER0}[0] < 0.5 and {ANSWER0}[1] < 0.5 else 'top right' if {ANSWER0}[0] < 0.5 and {ANSWER0}[1] > 0.5 else 'bottom left' if {ANSWER0}[0] > 0.5 and {ANSWER0}[1] < 0.5 else 'bottom right'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="What area of the room is the cat at?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What area of the room is the cat at?')=<b><span style='color: green;'>bed</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>bed</span></b></div><hr>

Answer: bed


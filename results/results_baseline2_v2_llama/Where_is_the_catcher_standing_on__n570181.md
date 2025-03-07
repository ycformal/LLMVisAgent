Question: Where is the catcher standing on?

Reference Answer: field

Image path: ./sampled_GQA/n570181.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='catcher')
IMAGE0=CROP_BELOW(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='field')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'field' if {ANSWER0} > 0 else 'ground'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Where is the catcher standing on?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Where is the catcher standing on?')=<b><span style='color: green;'>dirt</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>dirt</span></b></div><hr>

Answer: dirt


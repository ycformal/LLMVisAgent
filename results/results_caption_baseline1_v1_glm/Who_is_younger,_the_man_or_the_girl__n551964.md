Question: Who is younger, the man or the girl?

Reference Answer: girl

Image path: ./sampled_GQA/n551964.jpg

Program:

```
BOX0=LOC(image=IMAGE,object='man')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='How old is the man?')
BOX1=LOC(image=IMAGE,object='girl')
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER1=VQA(image=IMAGE1,question='How old is the girl?')
ANSWER2=EVAL(expr="'man' if {ANSWER0} > {ANSWER1} else 'girl'")
FINAL_RESULT=RESULT(var=ANSWER2)
```
Answer: "Runtime error: '>' not supported between instances of 'str' and 'int'"


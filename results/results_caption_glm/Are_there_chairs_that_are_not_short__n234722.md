Question: Are there chairs that are not short?

Reference Answer: yes

Image path: ./sampled_GQA/n234722.jpg

Program:

```
BOX0=LOC(image=IMAGE,object='chairs')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='short')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} < {COUNT(box=BOX0)} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Answer: "runtime error"


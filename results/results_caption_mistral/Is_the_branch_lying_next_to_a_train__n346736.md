Question: Is the branch lying next to a train?

Reference Answer: no

Image path: ./sampled_GQA/n346736.jpg

Program:

```
BOX0=LOC(image=IMAGE,object='branch')
IMAGE0=CROP_NEXTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='train')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Answer: "runtime error"


Question: One image contains exactly one adult wild pig standing near at least four baby piglets with distinctive striped hides.

Reference Answer: False

Left image URL: http://www.zdravasrbija.com/images/ZS_ds6.jpg

Right image URL: http://www.jagranimages.com/images/07_12_2015-07wildpig06.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many adult wild pigs are in the image?')
ANSWER1=VQA(image=RIGHT,question='How many adult wild pigs are in the image?')
ANSWER2=VQA(image=LEFT,question='How many baby piglets are in the image?')
ANSWER3=VQA(image=RIGHT,question='How many baby piglets are in the image?')
ANSWER4=VQA(image=LEFT,question='Are the baby piglets standing near the adult wild pig?')
ANSWER5=VQA(image=RIGHT,question='Are the baby piglets standing near the adult wild pig?')
ANSWER6=VQA(image=LEFT,question='Do the baby piglets have distinctive striped hides?')
ANSWER7=VQA(image=RIGHT,question='Do the baby piglets have distinctive striped hides?')
ANSWER8=EVAL(expr='{ANSWER0} == 1 and {ANSWER2} >= 4 and {ANSWER4} and {ANSWER6}')
ANSWER9=EVAL(expr='{ANSWER1} == 1 and {ANSWER3} >= 4 and {ANSWER5} and {ANS
```
Answer: false


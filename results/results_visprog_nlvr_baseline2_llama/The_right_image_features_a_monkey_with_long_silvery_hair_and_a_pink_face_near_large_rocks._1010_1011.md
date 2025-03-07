Question: The right image features a monkey with long silvery hair and a pink face near large rocks.

Reference Answer: False

Left image URL: http://s3.amazonaws.com/medias.photodeck.com/058f9070-2a9b-4570-a272-6a290243468d/Brett-Cole-South-Africa-00053_medium.jpg

Right image URL: http://s3.amazonaws.com/medias.photodeck.com/013ca52f-f79d-4ded-918f-b71db871148b/Brett-Cole-South-Africa-00051_medium.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='Does the image feature a monkey with long silvery hair and a pink face near large rocks?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='Does the image feature a monkey with long silvery hair and a pink face near large rocks?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Does the image feature a monkey with long silvery hair and a pink face near large rocks?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="False")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


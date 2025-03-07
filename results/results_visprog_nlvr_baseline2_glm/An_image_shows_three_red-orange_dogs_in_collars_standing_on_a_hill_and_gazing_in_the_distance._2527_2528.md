Question: An image shows three red-orange dogs in collars standing on a hill and gazing in the distance.

Reference Answer: True

Left image URL: https://i.pinimg.com/736x/25/16/b7/2516b700b213b882a7387bb2b4206051--vizsla-pups-aubrey-oday.jpg

Right image URL: http://2.bp.blogspot.com/-JB0H6m-YpB8/U0CcxMVyubI/AAAAAAAAH4U/f9fVyLWqWRM/s1600/DSCN1988.JPG

Original program:

```
The program provided does not match the statement. The program is checking if there are two seals in direct contact, posed face to face, while the statement is about three red-orange dogs in collars standing on a hill and gazing in the distance. Therefore, the answer is False.
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is 'An image shows three red-orange dogs in collars standing on a hill and gazing in the distance.' true or false?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is 'An image shows three red-orange dogs in collars standing on a hill and gazing in the distance.' true or false?')=<b><span style='color: green;'>false</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>false</span></b></div><hr>

Answer: false


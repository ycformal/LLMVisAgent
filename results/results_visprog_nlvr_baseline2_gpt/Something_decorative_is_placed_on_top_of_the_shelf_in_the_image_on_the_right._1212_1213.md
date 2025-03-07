Question: Something decorative is placed on top of the shelf in the image on the right.

Reference Answer: True

Left image URL: https://bizimages.withfloats.com/tile/57d4cfb39ec66822e83a9159.jpg

Right image URL: https://cdn.shopify.com/s/files/1/0191/2234/products/1304865832SC-655.jpeg?v=1418105252

Original program:

```
ANSWER0=VQA(image=RIGHT,question='Is there something decorative on top of the shelf?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='Is there something decorative on top of the shelf?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is there something decorative on top of the shelf?')=<b><span style='color: green;'>yes</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="True")=<b><span style='color: green;'>True</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>True</span></b></div><hr>

Answer: True


Question: The right image shows a man holding a giant cup of coffee.

Reference Answer: False

Left image URL: http://4.bp.blogspot.com/-QKVU84M1z9w/T0wbb4NNJxI/AAAAAAAACFw/ZSrVTlQsbYA/s400/decoracion_cafe2.JPG

Right image URL: https://bellnu.files.wordpress.com/2014/05/el-cafe-reduce-el-riesgo-de-diabetes-tipo-2.jpg

Program:

```
ANSWER0=VQA(image=RIGHT,question='Does the image show a man holding a giant cup of coffee?')
FINAL_ANSWER=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Does the image show a man holding a giant cup of coffee?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: no


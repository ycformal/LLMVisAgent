Question: there are multiple jelly fish on a black background in the image to the left side

Reference Answer: False

Left image URL: https://cdn-attachments.timesofmalta.com/69601a67d538189955dc6384cd1c9988491a52ad-1506059655-59c4a587-620x348.jpg

Right image URL: http://cdn.newsapi.com.au/image/v1/cbfea4019b5cd2ddcf2e53b5f296ebf5

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many jelly fish are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} > 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many jelly fish are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} > 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many jelly fish are in the image?')=<b><span style='color: green;'>1</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 > 1")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="1 > 1")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


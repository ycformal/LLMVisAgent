Question: There are only two dogs and both of them have their tails curled over their backs.

Reference Answer: False

Left image URL: https://i.pinimg.com/736x/b3/bb/1a/b3bb1a82a771068fa497c4f141114926--samoyed-puppies-animal-magnetism.jpg

Right image URL: http://www.dogwallpapers.net/wallpapers/beautiful-japanese-spitz-dog-photo.jpg

Original program:

```
Solution:
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is 'There are only two dogs and both of them have their tails curled over their backs.' true or false?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,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">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is 'There are only two dogs and both of them have their tails curled over their backs.' true or false?')=<b><span style='color: green;'>true</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>true</span></b></div><hr>

Answer: true


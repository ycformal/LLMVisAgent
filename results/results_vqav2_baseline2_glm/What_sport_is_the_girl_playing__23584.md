Question: What sport is the girl playing?

Reference Answer: tennis

Image path: ./sampled_GQA/23584.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='What sport is the girl playing?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='What sport is the girl playing?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABDAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDibe1JArShswQMjP1p9vCMDitGKLjgVzNjKq6dA33oYz9VFOOkWbDm1iP/AAGtJY8U8R+1TcDEk8P2Eh+a0hH4H/Gp7HTV0wSfYRFH5hG8Mm4EDt14rW8v2pyxFmCqMk8AU02Bl6pHcX/2xSFWG7I3wqx2gYxgA9u9Yw8H6aVBaNg3s3FddHoWrTE3d6o021gBYCSVcznHCYBJ59qjMVF5LyKlG25ybeD9PPQyD8qgfwbZ9pJB+VdgYwKYY/anzy7kWRxMvg+2AOJX/KiTSriM5W7PBDcxr1AwP0rsJIgRVGeAelPnYzi7zSJbyfzZ7je+MZ29vworo5IPnNFVzsLGtbz2jDgzr9UB/rWlFJakAeeR/vREfyzWRbzQ8ZtIvwdh/WryPbHrbuP92b/EV2/VoPoTc0B9nPS6i/HcP6VKqRn7s8B/7aAfzqgPsp/huB9GU/0FPEdqf+Wsw+sYP9al4WHmFy+ICejRt9JF/wAasWyTW86TRwh2Q5AZdwP1rL8u2P8Ay8/nEaWO2QyLsuYScjAwwP8AKp+qrowuWjda1rPiaO3vLaGWDJ8l4IysUadySf4s+/pWzrPh2TT4vtEEgntgQGPRkPuPT3rSsspHm2vDuXnZJjaT6cDiiy1GS+vZra5hNszqU+9kOp4yD7GuOMU5tSTTf9dDtqQ5oKzTSOP2jvjNRutWJotXt5pIWa4Zo2Knnd0qu02pL96OU/70AP8ASuj6o+jOO5GU44GaqzR5HIq011cgfPFGD/tQAf0qrLfyDrDbn/gJH9aX1WfdBcpPD83AFFK+oDdzbQ/m3+NFH1aoO5XgHAq8gAFZ0DVeiPHrXpGZZBp45qMfSr2n2yXM7LJu2qoPHepk1FXBK5XGKlR2tXjuNoJDZQEZ3EU7VbWW1f7TE6i2RCXjMWencEHNLaX9wLeNpLa3Ey5Abbuwp6AA8Aj15rmrSnNcsNL7s3pKMXefQ77TYBLEjOq5wOB0zWR4onaw1Sykj+YhGY88HnGKj0fXZLiJswsNjYOGwP5VW8RX730se5VURDjByTn1P4CuClmmFnVVGMrt+TOurga8IOUlaxmandDU7n7f5axmX5XUf31GCfx4NUwzL913H0JqeBRLHPCMb8CVRnrjg/pVUg16FCTacZPVO3+X4HHVik1JLRjzc3C9J5f++zUMl7c45mcj35/nWhYaNeakrSRqsVsn37iY7Y1/HufYZNX9+m6R/wAeMYvLof8AL1cJ8qn/AGIz/Nvyqp1YQ3ISZlQ6Jr95Es8Vp+7blTIsaFh6gNzj3oqO6upbq4aa4kaWVurOcmiub6y+xWhzdu/HatGJh61i27nArTgY8V3pmRohuKmhmkifdExVvWqiYPXJqwpFDs0CLzXtxPD5cxjOeGKrjIpm9cYC1X3flS7iqkj0qOVJFXbZpWeqafptkWuJljBYliVJGffApdQuLaZ3eJiS4XHoMfWuY1xLlNNjiDbIiy+YyHJwe/0ziuu0a209bK31G9umkZlBS1gOGyOMux+6Ppk18rl+X0IVvbSb7r5P0Pdx2MqODira6MzrXS7rVbhY7WF5XXk7DgAf7R6AfWtpLHTNK+a5ZdRux/yyQkQIfdur/QYFLe6zPdxmFAlva5yIIRtX6n+8fc1ms+a9ydfV8qtc8XW1mWr7Uri9K+fJ8icJGoCog9FUcCsmZxUsjnFUZpcVzt3AY8g3UVTeX5u1FAHO2pOBWtASRzRRXsIzLsZPrUyE7utFFUIlyeKceh+lFFRL4WVHc3J2KeGNWjUKFljWN/lGSu1jjPbkDp6Vk6a7f2XDz0yB+lFFfOU9qfz/AFPSrf8ALz5Em9vU1ICfWiiuw88glJwaoTsfWiikBnOTuPNFFFUUj//Z">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What sport is the girl playing?')=<b><span style='color: green;'>tennis</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>tennis</span></b></div><hr>

Answer: Tennis


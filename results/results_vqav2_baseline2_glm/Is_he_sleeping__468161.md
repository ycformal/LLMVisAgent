Question: Is he sleeping?

Reference Answer: no

Image path: ./sampled_GQA/468161.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Is he sleeping?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Is he sleeping?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABLAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDxW05KcCtWzwuqWhLyACZDmP7w+YdPesuzAIXrmtiwBOrWvzlQJk+ZBk/eHQeorKZvA9M+IWkFvC986DPk7ZOPQbc/zrw5jzX1h4y0SB/D2shoozutZADtHB29f0r5QUB+p7ZqMOuW8R1XzJMiJqW3he4lWGMZdztUe9RGt/wakQ8WaVLcKfs0d1G0zdlXPf8AKt5OyuYxV2YH8IpKt3dkbaZIhIkm6NXBQ9Awzg+/NRLbszgbSPmwT6U7q1wcXexD0pQCeQDirAheSURYO4nGK6mz8OobQFyp+vSplNRKhSlJ6HHEEdRSVv6hpiW42FMKT94HoaxpUKuVKBWX5Tj2pqSYpQcXZkNFOAoqiLGlZZGOcdq1rSQw31u6kqRIpBXr1H61l2gww6Yz3q6zbHVgMYOfrWMtTeGh9I+JtWQaHfmQghopRyf+mbf1r5TzsUjA5GOlez+Kr9P7AuNiICyuMgDupH9a8ebiReOxrDDScryZrVhypIhlkEghHlouxNuVH3uScn35pbdQZRmUxju3XFOdc/wj8Khxjrn0rrWxzPRl1I7m8vViikaVuAp29gMfyAroxo9tDCFcmSXALHPGfasXQfOW6Y2+PMK43E8D2pmq3Uq6rJ5cz7o/k3A4ye/61FruxrdKPMbVtp/nakrRrkIhOB3Oa6iHUJWs0hSEeYP+WmASVz9K4vw9fTSXTxvO29xgHjp3ru7kRWdtDcWmptKyrt2y24UKOmCwNTJamlOStc5nXEYwOChJYEBG65PbiuPnJjZ47iIiY4yWyCpHXjvmuy1LVLe11SSWU75AuY025Bb39K468kmvL2S4lGXlYsauGxnVd3oVxt7nNFIY268c0VZhr2NOMYbgVYkOVz9ORUcrRrKMYIx2pDOpTGD9ag12Z1+vagsulyRBwSwAwDmuKkGJk+hrWvLhpoY4zwWwfwxVIwZOSORwKzox5UbVpuTK+AKqMMhcA53HNX2t23bgRj0NVUj3CIHodxrZGEncIJEjZi6sGKnaynBVuxrRtbFJbdWnTdIckk9T9aqRQ/vDkfKOQa24Fwg+lawjd3MpO2iGW1l5EsctsBHJnAfriti603VL5x9snijjXkmIZJp+momwu23EZzz0FPvdRV4/Jt8iPHLdC3+ArJqUptI6EoRpqUjl9QsVkmZo8rg4GTnP1rKlhlibDZB7V0jqWOO+elVJ4flwQCvoa1cNDBT1OeZWz0NFaLw7WxjI7UVld9i+RPqHl0BK1LfSrq53eXbynb97ETEj8hW3ZeDri5SOSVZY4nG7oGYqOpCqcAe5NS2WjmnkZo0C8sOF9aljfjEqNG2ccjAPsPevRfDnhu0+03NtFp8V1IYciR7h8pnodoGDyeldMPC1hZxOAWeVU2xg2SFGKkEll6gH19vWsueMSp817NHjwtwRyuccn/Co2s0hQ/KFCjHHqa961WLws3ha3gutMUQR3RJMSiLZIQcspBHoPy9q4HXPBcb6QNa8P3L3lgMyPE4zIAOpBH3gPz+taRfMroScep58bcookIGwnoO1XIyNo49selVrtt0kapnbySR04/8A10+In5cHg10Um2tTCqknoXsp5CgZDEndk8H04pjnJ5PtzURYjr9KsRxm4SR48YRdzAnoK1Myu3XJqGRgc+9SzusfPaqjPmk2CI5ArNnHaikLZNFZlHrVv4dgv7lF1C4ubuPOSjybY/8AvlcD8811d/HbJYHgqgCr5SEIrAdAfYDtVG34uFxV9o0naKOVQ6NIoKnoea5INukzeekkZT30VmYb2a1eFQTtTaAAegbAOfoa27NzdyBjM6GeIRupf5iep7d8n61sa7Y2sNvBLHAgcbgDjPUCuKuJ5YLqfypGXBwOegB4rlU29zVpciaMHxPpwudUbTbLMsKcDBJ5759Pxru/DUU1to0FtPO0zLwNwwFXoFHtWcttDFYWxjjCl3YNjvjFa0p8rSZWj+UiI4IrehJuTl0SMqjulE+fdTaFtbvFgAFuk7pGF6BQxxTVOOcVveMrWCHUbaWKJUeUN5hH8WMfrWEv+qFd1GanG6M6kGmJn5T7jNJuKcYNIaicnA5rUzG3Dbhtz26iot3y4PWhydhNRgnYvPaobHYGbnpRSdetFTcZ/9k=">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is he sleeping?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


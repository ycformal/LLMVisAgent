Question: A waterbuffallo is standing on water in the righthand image.

Reference Answer: False

Left image URL: http://3.bp.blogspot.com/-HaXgWOWhpWs/TuEwV3BDF8I/AAAAAAAABBU/rl7DFap60ZI/s1600/PH200910-123.JPG

Right image URL: https://s-media-cache-ak0.pinimg.com/originals/92/b4/d5/92b4d5845acd069e0236ad24a56fac1b.jpg

Original program:

```
ANSWER0=VQA(image=RIGHT,question='Is there a waterbuffallo standing on water?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=RIGHT,question='Is there a waterbuffallo standing on water?')
ANSWER1=EVAL(expr='{ANSWER0}')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABLAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDu0iCKBwuakyMcGuEh8dsttJNdWm5hjaEOA3+FVD49v55o3ht44oVOSuc7/Yk9PwrreJppXbOflZ6OPwqhLrOmWMqW9xdKspz8oBJHPfHSuJj8eakpG6O3dfYZLfTFZWo3QvpxdTFxLneG29Pr7e1ZyxkbXiNQfU9Tm17T7UxKVeTzEEgdFLDae9Pt9csZbtbUlklckDcMdK87h169mig/fRqIV2fKFGUHTOPrT9S1gXVuIWEjkPkyBMBTx1I6j6elY/W23oUonqblFHcmmGb5cZ4rya312+jsBaxzONrEqyuQTzkD8/581KvibV1bMtzK5IxhgOD9MVr9ch2Fys9RVgT/AFqdX28gH6mvItP8Ratp0dwqTRt5jbnaT5jnGM+1WdM8Uawln5Ul6ZUX7rEAtnOeT37j6U/rkOoKDPVjMzn6U15C3XmvNYfGGpQQOZJFZRIWzt5YHt7elWZvHV1JHNHBGiSE/uzjcBjqDzS+t0iuVndFjnkiivPJfHd7E+17SLOM9/8AGirWKpdybM5oaTKUBfY6gcpupWtFDeRIhKBN2wnj9K6BUt2GAXyOwGa4/VdO1N/EtzHYxNOko2gMdvyhVJ5PbLdK8alJydpM6eRI1bfTIZbXzYHKyHtjOeOmO1SPpgym55g+ADg4qLwzaX13YzajPK0TeaUityDtUA4Y9eMnI/Cug0uC4ms3+1S4uFkKsxUbcZ4x+FOrzX913sNUm9jJtraOJ9rDagyCOv8AKsnUwLqZ4DfR28QcDMiFsntwOcDkn8BXYXKQQ25QzbpGU7cDHOP0rzq8jure7EM65jjXAI+YN68+tVQhKXxBy8j1NW9iubK4NrLMskkWCjR4KMp6MDnnjnpUmnW8vnSLJcrKQdyqHIbafUViRW92BPN9nzbRru858gDPYepqjYXzxaqLwZ37wRjuAen5VtOleOmjCST2OymtI23qYGiJHJPOTWne+HpNIsbdzLbu8qhjDGcuox1I7j6V0miaSdXuFgYtDFsLu5TlRwAPrk0q/DS403xJLf8A2mOSyYHOwBXCnsSfp2rznXjFPnZUaKfU4aOGVASI34U4+Xp+dSpbHczQ25WYEZY8A+tdRfWkmn38ttIQ2w8N03DqOPpULeXtB2856g0KpF2dyXBJ2uYMsdwz8OuMDGVorYVmA+d8H2IorTnXdD5Y9xAAD80rLj+82MULFAzFvOYnuVbFdK2k2zAeZGRQmg6ccnyck9yxp8p0qlbqcwsNpb52bhk52m4wuT7VNuiU+Yr7HP8AEkpP/wBat99B02Ah/ssr5GCVQMfzFVbnS7NHwlpcMexRN30HOMU1dO6K9n5HEXX2u3W/kmvPtSlcwsV+ZSTwD+Qrkl16+tL7yVvGEQ4cgA7m7kfj/KvWL3wnLfJFLFqT21uyr5kBiy3rg4PH/wBavGL/AEi4h1Ce3TLywuyPEeHXHXjv9RXZTmmtTnqQcdS5d6vdXzOsksk0JGEVm6E1Potib/WrO1HG6QAgenf9Aa6Hwt8On1bSrbULm+FuJC+YWTDADIBzn1APTpUngfQLy11q+kuiEntMxhj8w3E4zgdsA4+tKc1Z2HCm7q/U9Pt9SmsVZILPzTKwDt5mCijnPSmaj4tspLLbFcSyzMD/AKOFO4H0IqFbiTTYzezDfBAA7FOp56VyK+K9C/4SC6MltdRmWQvlUUgL6deua8WvRk23GLenT8jrlKF7yOnMs+qAXGpW0EVx91VTk7R03HPX6cVE2nw7cBCB6KamhP2y1iu7ZpBBIm9N45welKCznBZh7NXQqcbWsDjB9CgdKhzxvorSEcg6NxRR7Ndhewh/Ka6o7KoUEOfWlaOaJjkYPrtpBC6yNhCT79KUyMh2lGzn0/rXWK4fMB87YHqRTGjmdcwzRbgejdxU7Ojk7omb2JOBUXlSE/u0JPYAZpWGmHlXbEiNgwxkqVzXmvjrR511e+uo4hGbjT8zso4IDKD+OMH6V6SqShvmbYe/P+Fct4n0q/vLa5nllgeOONthiRt5yMbSD6irhKzM6iujc0izl0fw7ZQGa3gjit0B3Lgj5Rnkj1zWY2oTeH7m7uBa2ccd4fPeaRmJkfHyggdDj8BXP6ZBqmpXlsJ1muoIWUMXGRtH+yTiuril11ftBl0uB3ZiyKZAufTcQeTj0olozKfNKyiRXF9qWtaHqsFxZSROiEjy1xHs2hh1IJ6HpXma6ER4Wk8TtLu3XDL5AGDtJ29frzXf6pZ31xdWF/qWn7IjA8d6qsR5Cg8ODn5jjOAOax7eeBvAdxYLLtUSsqjOS+H3E47U030CnHmVqnQ7rQrX7D4fsLbzjOI4FAfbt3AjPT8atyxRyDDxqfciucg8SxkJFbW00iqoVAAM8DA4rYs57icN59pLAAM7mGQfyrNxfU6U1axKLSEAAJRTxhhkLn6UUrFDbbUZSq7nOz+6gAzVz7ZFcjy5I0VT39K56xJK881cZiOhq7mVjTaLZykm5cYwDxSqExxz6iqdu7KysGINaJRfLBAwfUUh7EQQZGAOajkRDlHCkHswBBpIiZM7jnkVMVAJwBweKVgIktooxhIgAT2GKk8tcYYAY9asb2EQ56mkdFGMDtRYdyu6LJD5Um1426qwyp/Ooza2ywfZ2toTF02KoA/ACppSQRjiq5YkHk0irCRCCMrbQfulQcKFwMegq2i+WeC4PsapSfMgBGaWN2VFwx7+9FgNBpwDykRPuo/woqorEgEnmii7CyP/2Q==">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is there a waterbuffallo standing on water?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="False")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


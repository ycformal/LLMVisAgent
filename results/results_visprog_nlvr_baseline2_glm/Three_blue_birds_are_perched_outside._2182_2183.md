Question: Three blue birds are perched outside.

Reference Answer: False

Left image URL: https://i.pinimg.com/564x/95/e4/b0/95e4b08728edb98c57e0fe3e518703db--blue-macaws-parakeets.jpg

Right image URL: http://farm7.static.flickr.com/6216/6222141051_2fbc230070.jpg?__SQUARESPACE_CACHEVERSION=1318066650207

Original program:

```
The program is not provided for this statement.
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is 'Three blue birds are perched outside.' true or false?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAA4AGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDF0q+S6trb7a0UKf3SpOeowSOg6Va+02sKCdYY53fbGWfOIx647jPNYtrcX+lXK6ho5ZH2uERlU8cYzu+uKXT7mVYcXHyu3LEdM96yUU2TOTjrfY201N7K9IbmGVMruAAGDw3+fWmw6qbWOV4o3kkMgKkrwSR0/M/rVDUkNxaW9nbcK2VUkE5HUH8Dx9CKdFcXFhogYx7bjd0B4U8f4ZNZqnrZj504qS6mu+o3iBYxbRpM2WnUr8gUf1x6d6uFo7bTJLyWzluLSZFUSRt/qkVsgH1+bH5Vx1/qlrqVysmnmeNIoSZzPLuLSjljgDCjJPFdfpV2tv4e09J4Wjjvo/LO4f6xAQS554BJAz39KzqR0uE72NSCPSr9GtrZpoRDEpBlcglz8uNwPI5yRnpUNul2Yp3d/M8lFUu6bgqZz8rdGHB4qC2u2u4Rd2C4sQ7ReUz4ZgTggA5OTn0zW0txb20lxChWKAMgLE4UBuvy9zzk1ySbe+5nvqzOsJrqeeOOAxJbDMSrvIyAeTk5yMc/hW0git5lwXIEg2kPwT647VnwXMFvCi3LQieVcfIfujvz3z64q1qdlILJbrTZUYx/M8bHO5R97B9frXdRoKMOaotWZK85e69i4dO8n7XcJNHJDKikrJuLlxnr2wcjgda6ZLmCW3itpyPPRF4+6wIHJA9K810/VJdc0y7gtLiNJsKCysSY8kYb064rSsGM2npLqEEi3aOwJxtMDdCowfmUYzz69aykuST5UdSulqdXFH5bzMt0GSR96hlJ28DI59wT+NFYukazqktgssUDSROcxtjquB05oqI1NA5jyWylmltLSSbKyXDOQT2UMQP6/kKk+zxoWbeDz3H8qy73UTa3NjChDJFEIyB0DAD+uaS4vCznH3WB7969Knsjlrxac79f8jYumuDpBFo+xzII1cckA+np0/Kn3MfmQkNhkZVYDqxOOfz61StbiQac4DYO5CB9BV3z4444iQMbRnmrqRV7nPhpSUeXtf8AQrSWUT6TcBFCsIzIGOcg8Z4HfgV3MEZufB2ly+SgkitYkMyurDkjkr9cGuTsblJ7loCoCMrIc+pBxWFp9/rF5aW0UmvXtrZxxqqIkIHA9wOR9cmoqUHPSKLw2JupKo9meo2v9nW9ydMMaQ+WimeSeLa4cZyARx6fXJqtruxr42VvKbmUAt5rYAgU9NoAADdh9PzxNKeSBHSPU5r0Plmnmcsckjr746CmnVZbASEPv3uWYsPvn/PFVRwcaXvbsipXcnyLRFxI9OsLVljtYhO2d0pXLtn1Y81c027mEG1p9kLJhyDnHrXMS341C4zvwTyRjnPpW7OY7Tw3ctIqllg2gerHj+daVY3Vh05csrjfCtjZSXBsoIZ4UEyM8c4yZI0JIGc8qTz/APqrsLvQ49MaWW3XdYXC7biBmJEZ7OmenUgiuE8J3pN+jl8MvA57dTXcXHjGRfEa6fLLBJbPbriMRjKuSep68jtXPOnTbZ005VGrsYrmBRFZXTG3QbVHlg7fbpRXMagLWLUrpBNdMolYL5ByMds+/wD9aivNc+V2t+Z0XfYzp00XVFV57O3aQ9SkeG+uVrMj0O21N9kbiI8sDuwoGePxrB1K2Mev2MEDsiQxxrt385Lc/nWa+salpksjJcMyI5Ta/wAwwM8V3QclszWahLSSOov9E1jTtLuGitzdB+N9s2/A6Zx1BA9qoO06RPG28hDjJBHP41rReKobGNRNuKPEsm5UyAGJHb6U7+39P1LUY3F1E0YAUgfKT36HnNaOq27SRhDCwUfce/6mVpFwftSuSDlq19M0i1TQrKWWZ1nkQjywEBHXB5OSOBV6TR9EuLgyQyXFrMf4sB0z7j+orIuLq6a5l0wvbpa2x8tWDopYL3Aznn3/ACrT2spOKpu19zjeE9lzymr9h95OLaMW8UmVXq3TcfWswXU8txHAMMW+6pPT6+1dBZ6FZ6jAqT6zpViWON0kqu6j/vrH61b0/wAIeHdO1H7Td+KbSZCzBAkq9e2Tv9PbFdNXExTsjChg5WvYqaZpMUI3yTqzZy45AB+vp71b1KRtRj+y2+XjTlWIIV3A4BPcf1FdZJH4T+z+XDqtmpAwC88RBPvzkj8a56a90pYpVku7NnCNhhOhCkHAXg/MDxzXBUr82x6FLC21ZV0fwvrOTcTQKiNITE0kiJsB/wBlSWOewOAKZP4butM1X+0YbWSWRm3neynYSOT94k8/l6VuWOvWVnpv2RdTtGUMDHunTIHU55+tc5q3jaC785IbmAjG3zHkGGyegAP6muN1JPSJ1uhFL3nY6WwfNlHuuJkk/wCWiTWyllbuMnqO+feiuUtNXhuoBJc6jbRSg7Sv2gDp3FFZ+zn/AFcaVO3X+vmV7y2jgkaYyIbmW4VzEXz5a+xx04PWuW1uydr2aRQhhZ8As+PLJ55FFFdcW00gmiG5kH/CM2hwN67YSD1O3ccj2+asRXKgnvnkUUVuYs1oPEd7Z7REysFUD5vX1B7Vzt9cPdX08743yOWOPUmiiqa1uJaKxXzRmiigAzRRRQAUUUUAFFFFAH//2Q==">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is 'Three blue birds are perched outside.' true or false?')=<b><span style='color: green;'>false</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>false</span></b></div><hr>

Answer: false


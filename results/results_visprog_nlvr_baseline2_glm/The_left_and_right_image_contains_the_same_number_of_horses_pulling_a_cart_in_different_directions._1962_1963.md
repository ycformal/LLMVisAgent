Question: The left and right image contains the same number of horses pulling a cart in different directions.

Reference Answer: False

Left image URL: https://o.aolcdn.com/images/dims?quality=80&thumbnail=1200%2C630&image_uri=https%3A%2F%2Fo.aolcdn.com%2Fimages%2Fdims3%2FGLOB%2Flegacy_thumbnail%2F1200x630%2Fformat%2Fjpg%2Fquality%2F85%2Fhttp%253A%252F%252Fi.huffpost.com%252Fgen%252F5321554%252Fimages%252Fn-CARRIAGE-QUEBEC-CITY-628x314.jpg&client=cbc79c14efcebee57402&signature=096d9c2b91e6d51a66d60f1621ab0399d039c1f0

Right image URL: https://i.pinimg.com/736x/5a/c2/2b/5ac22b57ac1ca1781a52da55917c2299--dream-team-teamwork.jpg

Original program:

```
The statement is False.
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is 'The left and right image contains the same number of horses pulling a cart in different directions.' true or false?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAA2AGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDmrF18PzNqFyBKlvJHLDErYaVt3Az1xgZ6VasPEZl1SN5rYW/lmW9UNchAS5B2Z2nnPbj8KyZdQ12W2ht7q1MO2YSQNNEVwRnIyeuQSPasO512aL7PHapKoVfKKM3mAkMCCBjnpVQhBRYnuen3eh3mo2kcL3jSzSxC/UbFCFTHuKr7gDGenFc7o1pcyXGpFIJZkgaEqF+YLzz249a42XVtVWXyY7uZTGCSu7AHUnjp+FdxoXjM2Hg9rWVpIbm6jfy3UdWyQzE9uoHHpWU3yPTuaJcxa16xjltkWOZSyv8AMowzAnOMCrbL5mlCOVIZI59PDrsY5OHCDKkDHOfb0rPi0uW58OxX2nSQNIQDISSC5B+8GAx1yMGrFksttapHcbPNWIxkoc5UsG5Pc5z+BqYzvp2BwSVzmtWhSHRJyFA5j7f7a1g6Xpd7qFsZrVS2x9jKQfTtxit/XSToFx7eWfydauWFxc+G/DcatKnlyFi8cUmGjLgFS+ORx+NddR8uhzU1zHIXDR2OpzQh2fO0lguBk88e1dN4Q0m41N7m5tvKEUMgLyytsAyOBn1PpVTVNOS40+2m2M0skJlDjklgeR+Rz+Fdfp0yeGfB9nZ32nxpeuDcR+Ych3yM7h2IUjk8AeuKwk7GyVzG8STyx2V55ocLOrNCGHVAQuQPQkGqvheC2uVxLL5UxlISQnAAxk+1d1H4Ytdf8MwaXqKKLoLJ9nuEyPKk3EqV/wBkjAx6AV4xBdzw3r28jNEisVIPJ3KcHH5Ur3QrWZ0+qWtppmqXNpAt1dxo5xIJBgZ5xk9cetFczdz+bcu6ngnPzYzzzRWivbcmyPUPF1rNfWsEKiTBbazIpyoOOf0rn7jRfD9mkUq3s1xPBH5YgaLYrdct9ST61dPxo00j/kB3Of8Ar4X/AOJqKT4u6HOmJfDk7H3mQj/0GpuaFLRdCj1GUm4lgSONl/1g/wBd0yCRzgY/pWvoOjA+IZ4BLbBLFiqx7ch1Y7uBnjqDWUPiZ4bXO3wsy5/uyoP/AGWmj4m6Ejh4/D9wjDoVuFH/ALLUtXGnY7AsEvY47K1RtME6291IuF8yUtjIxjhSBk+px2rd+xoi4SSdB6byQPwOa8wPxO0j7IbUaNdCLdv2m4U/Nu3Z+7681eX4x2QHzaPcsfX7Qo/9loSXYGzX8dW7ReE7phKrEvGMGJAT8w/iAFefaPdTX12ttdCI2yxF2XAy2AByR6cVq+JPiTaeIdDm02LTZrd5GVhI8wYDac9AKX4bSqt1qru6LttlDMwwFQt8xz+H8qrSxHU0k1i0R7GBY95hwWCAvgDnoTjkA/XNbepG11y1/tHUdcSb93+7Ea5ZyzHA6jkE4xjgCuejkhsbAT26wuVcqXGNxQEqAQfXA/PipvCVjYah4knt7hgbW4hZtgfA8wsMOB2IBqXawztGuZdKt9NDXmXZwoQRqoA4zz16nGabP8KdC1O8u5Y7m6hv5ZPNU7g0aqxJLAbev41Q0jSrG20waldQyalel8xJcTEosRLKTj+9wT36cc0aVdy6T4lcxXdxc6HbOXRw25oeNuCOrIMt06fShWBp7nl+tabNpGtXmnTupkt5NhZejcZDD6gg0VP44mSTxpqkkJV0klDgg56qM80VopE8pxNFFFSUFFFFABRRRQBYsoXnuRHGhdiDhQQK7/wy/wDYug6p5hhW5umjSOEvuJA3ZJ9uf0rjfDwzrEY/2W/ka6OWRTqUaCPLFCC3t1/pR0FfU0dLiaIu0s0TBss5ZPbPrT/DsttZ+J4bm4REgMDtIqhmALAgAD26+2KkvLeEeFLiTdsk25LjkjnpWYL2TTPENpPZOvlRBZVaTowIxj8jUW7Dv3Ov8FahMZ76KWzLQWQYxMDx98gkgnp1P4VgWWu3z6ibZoliVrgLI3RmwcYz2HHaui0eFpbbUJYlKQyyROWAzyQ2VHHJBBwPeuWhDza029UyJmfkgZAz1pOw1cy7037X9wYTd+UZWKBIyQBknAopEE5Xofzop3Hc46iiirJCiiigAooooA2fCw3a9CPVX/8AQTXol54ejiLXKPmUJ36dPpRRUPcZjyxNPaPCxG1x37H1qjJAWO1nzjjpRRVxIZ0enahNpnhmO0gdlF1cGRz6bcdP0/X1rDuYc3Dvu5bqfXNFFKO5T2IxF7n86KKK0sZ3P//Z">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is 'The left and right image contains the same number of horses pulling a cart in different directions.' true or false?')=<b><span style='color: green;'>true</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>true</span></b></div><hr>

Answer: true


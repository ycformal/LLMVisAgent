Question: What is this man wearing on his head?

Reference Answer: nothing

Image path: ./sampled_GQA/132644.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='man')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='hat')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'hat' if {ANSWER0} > 0 else 'no hat'")
FINAL_RESULT=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="What is this man wearing on his head?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABCAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDwSJWkJGSQqlsV0fhpcl1ZsLjcp9DXORNLESyAgkEHA7GrlnevCw2MAw4weKiRcdT0VWku3aWW9RJZDlyz7S3vxW7b2MFpcWd1JqVt+8h2P8/Kn19+xrzrTpnMY82TzJGbJ3H+VdH5JKmJyDtwR7ZGaz5m3oaWSVmeoahrOn3nhxrZtRtftM+EkIbgLnn861tIk0+5uY4bS4WdlgEYWNST29vavMbPSo5dOaXePMVwBHjkj1Fdv4ULWNxHcIPnX5WHr71tGXMzCa5Tlvil8V7mDUJPD3h8GH7K2y5unX5mccFFB6Adz3rz2y8SagZFn1fzns2YxC6RB+6YjGffHpXWeJ/Cq6Frv2LUP9NN9IbxbjaVPzscg+46enSpRpenw6etgsKtDk7hJyWB/rmuepX5XZo7KeEVSF77/mVtH8F6VeXUUuo6zD/Z8YMj+WhBlx91B9f5VoX2h+DracS2VnIjod6M0f3SOf71UrWa0s7ZNLQlhb7v3pPQZyFJ7kVbM1teq/lTRyBDtbYc4Poa7lpFO255UotScb7GTd2dk53JKzDGTlsHNYU9nEWJYSADoQ9dFcWj4JXpXO3xeN8FTU3GomVPZxLLw7YPPWimSySM/wB09PSii5VjVXTI5kzCYxxjDDNUryxgZjbHHnjkFRn8K6iKzgiwZIWEf+ySK3ZPC/hqS5S70tLuM7eTLLk7sfN7V5ax1K19T06+FnR+Kx53HGYLTyJYVHYTAHcD/Wg2eoGQ3NjeLM3cxyhX/FTzXol14XtLu18iWe4VdwfgrnP1x71l3+nw6ZbCAt9phJ6TKCRxxg9c1cMVSntuc8U20kdd4ZVG0azlvpFS4lJQh8Kd3OB9SB0rZuNWttKcgRGRhwQoJC49a8z07SW1adLWJxDEHVwrOxQMOmM5wfcV2a2up3rfZEvtPFxNIxLOrArzgg4GO/atYtt3RUoW0Zd8V3EHiHTYrq4UxSArFE0ec8g44PHb9K881NNWsJYVeJmQnaJgflOBk/0rsLyK70mXU5L6SMWdsZFVUJIPlkHvyOcfXNcjq3ie08Q2umGMNbpbxs3kysM+Y2N231QY4J9TVUoe2qLt1E6zo0nZlYtb20ONzu5+83HJq1oWrwfb1t3iVVnYIWPPzdAf6Vz95Pk4BqokxSUMpwy8j6178kpQ5WeEm4z5kexSeHr6VCqwMP8AgNc5rHg3UwpYQFvpXqaeJtPm0+3uTIv72JJOWz1ANcxr3i2H7luN3vXjWZ7F4nkNz4a1dZyPs7j2waK6q71/M2WznHailqL3Tnz4oVw6iRPKQjHmcFvbvWhZ+IGjnlliZXtsBvKDAFfpmvNnuC4HqB7CvTfAmm2F3oSTXNsJJGkO9mY8jOK8+vhqcIao9GNX6xLlqPQ0rXxBFfMVTIP8PzDn6CsvW73zYw6MkgEijAG7HB5q/wDDzSFgu9SlnijcCTEW5dzIMn8uMV0uu6Dp1zod3bxkWcoV5hKSVVsn7vHbGffpWNKhCM2oswjT5WpHB6RqCRSpcPek3IlVFi65GMkn06VYj8RXyahDJFax7InJP70jv/u/yqG3tre3tEsbH73LEkcye5+nvVOS0v8A53WF2C9drAgV1e0a+FG/sk3eb17HQ6jrn9p3N5NLuAuVdWQOcAMQc/XjrXPyixtovLSGPH+7z+dUU+2TyFIoZGYdfb6mqesC5tHjhlkj8x13MEbJUds+lRChOo7FSq0qMXZFK5u1+0yLFnaOnoDTI5jnrVQ8U9GNe9SvCKje54lRKbcrbnbWGsMumRRlzlVxzTrVLrWZpktSC0WN+9tvX0/KuNF2qyrG8jhR1wcce1WdN1650m7eSJlCyDD8bs+nU/yrhqc3M+U3jblRvXWi3qzkNNbg46GQ/wCFFYVxr13LcyPG5dSeDtNFR+8C0ShBaxlwzoxTv2J/wr0Hw9qUmm6UIobVAu7cobJNYekaeryxqqGaQkg8dPoP8a9P0bS4tPj81VkeYph2cgKfqD6elclepFfFr5HVGVvhPPL7UNQ09EaAzL85JSMtgsfXHWuh0RNZ1SykmvnjtoTHn9+7D8Tk8D26111pGmuXDWdvGsSQsWkmYfMuT0XHrya3LvS9F0/R7lb233223L73yWP1/AU4yvG7VhI8lsLiW2Nw1oyl5FEZJjOVPcLWTLd3sksdves32cOd3l58sY65IHXFGreIZo7gxWd4iQI58kBMyqO3z9/xqGDX5GeMSzymN5MImeGYYyW5wOfQc0QjLm12LniU4pRWvU3bC/spdMne3R1WH55BtA4Bx/WsnxNqEeo6HbvAwMMUzfeQKysQM5PU9vyq1aQaWIbyHVEaESsfKuoGIEbdfuD7wPpXI6tEbOSWxW53IoDBQ2Qe/Poea6KVT3jCpSstSgzg8U5TxnoKp0ZJ712e1OfkJpSHmcqcjFNjQyEIM8mlij3oxzyOKa2V4PXPWsW7s1UWkhyzui7Vxj3ANFIhi2/OHJ/2SKKQjsPMeOY+W7L0+6cVvWk8xX/Wv3/iPpRRXJU+I6IbG1ok80SHy5XTdydrEZ4qx4kmlk0acPI7fd6sT3ooq2JHmTxoZ1yi/e9Kmlij80rsXaUJIxwT60UVUTMihdmhuQzE4tsjJ6GueTlCTySDzRRV0yqu5BRRRWhkWbb7j/Wkk6UUVHU3XwEB60UUVZif/9k=">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What is this man wearing on his head?')=<b><span style='color: green;'>nothing</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>nothing</span></b></div><hr>

Answer: nothing


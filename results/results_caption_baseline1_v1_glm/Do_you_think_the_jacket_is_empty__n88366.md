Question: Do you think the jacket is empty?

Reference Answer: no

Image path: ./sampled_GQA/n88366.jpg

Program:

```
ANSWER0=VQA(image=IMAGE,question='Do you think the jacket is empty?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAA4AGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDnwiK43YBzxmrRYRx5Jx71xV/qDX1+ZkLKiH5Fz0x3qxd6nPfWUEZJB3Euw7jkCtnVTDlOtSQngkVXudTt7ZwrEk8Z284rk5Lidvs6+YRsQrvVuu45qxFp1ydUtrYjZLcSeUPNbaM9OSeOtQ6nYaibEmrxSXRt0BzyM9uKz5L8G78r+Ernd7+ldSvw21WO43MsbAK+4RSruzjK9fX68VU0XwhLq155U0dwkIRnMqgYBAPGTxycjNP2skHIjGhn/e7QSWABx9a0BcqqojH5nOB+Wat3Wn2EMkwOmzWM6RALunaQbh7Yxg9jnvXN3F5uiilT7ySZHHtV+2TJ5GjXF6jSSwKcPH1zVC7uVHcsfasW8lb7ZNKrEBmyD3AzxTjO7hjn5s/pVRqrqJxZoCZ3jZVXk9/Sq77kTJUjPenw3vlIMx5+UEmpmZJyrMThuMHsa09pG1yeVmXJII2AKnpn60VI2ybDMdpxjGKKh1H0K5TsLf4Sa8yFpJbGJ8/89WbP5CtSz+El8UjS71G1TbnPlxsx/DOK9Q2SZ4kH/fVTJHKR/rFrmsizgLT4S6fBJG8+pXUwXBKrGqA49etXr/4dWNzf211bSvEIn3OjEvvBOcDkbfrXarGR96bI+lVtQu47CwuLt3YxwRtIwUZOAMmiw0xzpKU2h2UdOD0qla6f/ZlhLFbzbT87h5TkAnJyfYE1W8OeKLPxFaSyw745YjiSJyCQD0bjsf6Umu6KNXt2e4vr+3t4433w20gUSJj5gRgkkjikU4tO0tDz7SPF+m6rcQ2XiO1SednMUV+qhdwJwNyjHGTx+FbafDBbfSLyyj1FpGkZXhMkYxEwJ69c9ccV55pMf9oeOtKuLCzbTtPuLpTa+adw2oecMepyOnqa978yRR8zcetCXcqq4NrkPLdf+HuqsbSWzt7aeTyBHdfZ8RAuP4gpx2x06nNVdG+HV1eG4XVLe8s1RNyYVW8w9gK9eBJ4wfzpSMDtmnYyPJLP4Z3d1FKHukgKEKokjIYg88jseazL3wRrMeoTWsMUc3lorq6vtD/TOOfavaiyjPTmoisTdQCPcU7Bc+b/ALPcZOIJDgkHC55HBor3yXw7o00jSPplmWY5J8ocn8KKLBc3lWIMMkVYV4lGd/6VQy3QKR7njFSKDt6jFAFzzYyPX8KoatOBo98Y0BIt5CAen3T1qTYEUliAMdWOB+deWa9d3d74kvLL7SX3ZKww3G9dmM4wOCAPUVdOmpu17GVWq6S5krmH4M1W4sfENqsMxiiuHSKTK5BU9Mj2PNezpCzqy3n7zkgbSVBHuPWvG/C2k3MuuxyPADFa3KOXidRuHUAEnBP48V7HHfajLqEVsdGlWE5LzvcRnYP91ck5rJQ5NLno4vExxLU0rPrcWHTNJhgW3TToFiU5VBEMA5zke/A59quM0AP3HqbkEBolBHT5SafnAysafXaKZxlQvGB8pwPxqMoWbcDk9qtvK5GAxx6YqIq7kZIJPQd6YFV4gD8w/WmlQuMYA+tTuny9eagMWcdKYiM9eFB/Gin7B7UUARaNqlpr2nx3lpJmNuGU9Ubup9xWf4wjuP7KhgtBqRkknGWseGAAP3uR8p9u+KKKlO6Na0FCo4roc/aeBZdQgR76EsGwSL+7dm/74QD8i1dpo/h/TdK0z7IlrbtGc5VLdY0OfUck/wDAiaKKZmaixxRQ+VHbwxxAY2KoA/KqyLLbL/o0hRR/yyckr+HpRRSAnEpcHflWA5Ocj86YruxJRDj1oopgIynIH3c9c1IMRw7mTJ4PHaiigBjJDksw7/dGaqXG51BhUjLDgjFFFAh6woFG4jdjnc3NFFFAH//Z">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Do you think the jacket is empty?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


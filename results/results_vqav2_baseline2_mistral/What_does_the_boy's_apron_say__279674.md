Question: What does the boy's apron say?

Reference Answer: chef

Image path: ./sampled_GQA/279674.jpg

Original program:

```
BOX0=LOC(image=IMAGE,object='boy')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='apron')
IMAGE1=CROP(image=IMAGE0,box=BOX1)
ANSWER0=VQA(image=IMAGE1,question='What does the boy's apron say?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="What does the boy's apron say?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABCAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDx3QONUdfWNv5iunC1y+jfLrqj1DD9K6zFcOIdpGsdjEvl/fXQ/wBqM/oapWsIklYSK/l5BJArUvV/0q4HrHGf1IqfTrcGx3AAO2cEjNUqnLEVtSpdWtpJKGMkhwgWMhcEY9qypoDAXlQkKfUA1110dLZ7e1ktIliCM52Eht56/rWFfQoZVjjUKpIChjwPrRSqX0OipTVrmDaW8lzexxIpLE5x7V1KWqwOgCMuc5yc54qxpnh1dPmNzcskkwxsCZwn+NWbtcyxH/e/lSqV1J2jsYKNtyps4ppSrG3imFazUhlZ1+VvpWFqa9vaujdfkb6Gue1X5Wx6jFb0ndky2MeilxnmiuozNbT/AJPEMPpuI/Q119cfExj1yHB48wV1iNziuHErVM1iUrvDXk2CCPJAyDkHDVes02WUK45Kis+1Uz3QWT5S8cgOB/titzToo31K0tWkOx3CHHZepP5A1nJXagPzKms2dza2cMiwq0jjfgnlVPT8/wCVReHvD02oLqGo6pL5dtZQGQjHUnOP5VsXmqG5vZpyNqsxO3HKr0A/AYFWNXuTpHg6G3jX/SdZbzJCTjZGCNoH4Y/M11KMYxSjuQ3O+pz1rM1qEEk0siuu5lkO4oO39OKuXCbpYFXkscD8apvbTOpiaZkDDM8m44C+i9gPfvUNndCext9pYsoZck9wDj+lZYmmk1JCpyb0ZbIw8iZB2OyEjocHFMIqDTmZ7FHb7zDJ+tWTWEtJNGnQikH7tvoa5vWB8/4V0so/cv8A7p/lXM6yCGUegGa3ofETLYyGyTwMUUHr60V2mZbgnMl/byN97eufzrs0PNcRZgtdRqP74PT3rtEcVyYlbGkBsgC6jBg7cxvyv1BrvfCdmNL0+e/usB5gGUsBuEYHHPuTXEi3TzI724cJbxBgc9XzjgV3sl7ZapaRrDdL5Eqggjt7Edq5bmqi1qyULFqyiS4tYWDthWK5YiuF+Il5HF4ks4mX93bwhBEp4A45z9ePwrurq4n0q0WS3tUukUBQysSR2xtAOa4ObQZdb8Qyahf2j21srLvgkXbkAcKo9O+frWlJqDc3sglFz91HOT+IZbiKSOVBGG6eW2Rn3zWjp0H2a1twfvlix9sj/DFU9fsLabUiLG3WJF+/s+6AK0ycyq395lOPT5a1rVOeCa6mfs/ZyaJkjSGMRoMKOgpaDSZ4rjKGyf6tvoa5rWRlz9K6OQ/u2+hrm9YOXGOtdWH+IiWxjHg0UHOaK7jM9GTwnpsbK0SOJAQQXkYj8avxaPcpyn2XPrsOB/X9a1I85AwefQVzWo309pdr9psLq6KysSXQtEU/hCqOB75ya82TlPzOqD5XZaFm50Rbi6iNzqsJdT8sJ24/AZzVmDTns5y6yQTkjbtDhWz7Z61BL4yM00X2XQ7tQqbDsj+ZRjovGBjtUM8/iPXWhRdMjgiT+O9Ick4xkg9ePapj7S6vGy9QlyvVvU1jcXETBXtbqEngEDA/Q1Xv9QMdv5cW55G4zknJ9KfcaVq7zIFmgcKqoC5I4GPY1g3GoPZ3jxyeVLNbk7mUnaW74+nTNEk56I3pyhDW+pfttOgtYWlvVEkrdY2PA+tY99eQxTgxxbUDDhTT7jWBep8gKueCpPSm6HJcTamI7faF/idlDcd+tVThPWVQzqyha0RBqlu3dh9RVq3zcn90QR3PYV08lnay/wCstoH9ygqJNPs4c+XbpHn+7xUySt7pnFK+pysk6HzkVssmQa57UmzLn2rttR0zSrK1mlLSRO4OAHzuP0NcLeNucH2rqorW6Mpme3Wilf71FdRme1xfdFWVOBRRXlM6CVTmpABgnAooqRir95fwrzq1VWn1clQSJXwSOnzGiirh8LKh8Rz6EhpcH+A11Hg9RtuDgZwOaKK66vwsyW51B+6tMI5oorlRRw+uszapMGYkDgAnpxXOz9R9KKK7aeyMWU3+9RRRWxJ//9k=">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='What does the boy's apron say?')=<b><span style='color: green;'>chef</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>chef</span></b></div><hr>

Answer: chef


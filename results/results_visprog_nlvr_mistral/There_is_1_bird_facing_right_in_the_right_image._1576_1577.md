Question: There is 1 bird facing right in the right image.

Reference Answer: False

Left image URL: https://c1.staticflickr.com/4/3323/3411279617_3d722de017_b.jpg

Right image URL: https://maltpadaderson.files.wordpress.com/2015/11/pelican-hasting-harbour-15-nov-2015.jpg

Program:

```
ANSWER0=VQA(image=RIGHT,question='How many birds are facing right?')
ANSWER1=EVAL(expr='{ANSWER0} == 1')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABQAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDmFFPp6pTwlcZ0Fc5phBNXPKpPJoHYqhDThGatiGnCH2pMLFUR0GOrnlU1lAoAplKNtSsKbxQA3YKQx8VJxS9aAsV/LoqztFFUFiITCpVmFYonNOFyR3pEcxtiYUomFYwuj605bk+tAcxuW4a4nSGMbndgqj1Jr07RfCtlZQq1zEtxORyzjKj6CvLNDN9LdSzafGJLq3geaNW6FlHH6mvW/D2spe6FbXdyq27PGpZG4wSK3oxW5E5N6Fu60nT5ojGbSDaeP9WBXm/iDR4rJXuLWRWiVtroDnYT0wfSu61zWYoNMmltmWV1HCg9815focuta3puorqsjCdmLRoVA49PpmtXFTViFJx1Mx5hUfnA1TdyCQetR+Ya4zbmNDzh60qz1nGU0gmOaaDmNbzhRWX5x9aKdg5iLFRtkGpQc0jLU3IGKTUq5pqrU6rTuh2Ov8BXcNrcXplUuzxKiIoyWyegrtNCSZriV7hHR1+QRsc7FB4H/wBevNPD0pttatmHRm2H8a9ehXdMWJ/5Zrmumk/dFbqcl41kMckciEq0f3TnqSeh9ax4ZZbBUe4t5LeWVTw7Zz/UfQ102pxLLrls0mGSOQy4Pqqkj9cflXG+I7wyyu2c4yAfc/5Na8/LqKcE0c3KfMdn9STULDFSZph5NcNhkRpucVMVqNlosAwtzRSFeaKNQJFOKfupCtGKzAUNzViI5qsBUyNii4XNbTBnUrUDr5q/zr2CBsPIfTA/IV5BoREmuWSn/nqD+XNevW2PKc+5NdeHejGc9qcoF4zj+FJcn04Uf1rzrVpi8ynPDZb+ld3r0pjjdh0aJhn6uP8ACuC1ROIW9QRV1fhYSexnk00MKa5xULPXMmBZLio2cZquZDTd5p3AsZoqv5lFK4F8DIpOlXlszg/vYuOPvdfpSNZOEDboyD6OCaxdwsUc09QTVsaexOFlgYeokH9aeLNkTcXiPfAkBNFmKxd8Lx7vEdkDz85P6GvW4WAs2b3Y/rXl/hKLd4msxjoST+Rr0uRiNOZieqnAFdmGXuhscd4on2JChwCYUJH4sa5HU+YoG7EH+ldH4tYnXYYB0WBM5rmdTyLe3B7ZFaVlamyG7yRlOcmo2jzTyeakUDFeemVcq+UaQxVcIFRtijmFcplOaKmOM0U+YLn/2Q==">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many birds are facing right?')=<b><span style='color: green;'>one</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 1")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="'one' == 1")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


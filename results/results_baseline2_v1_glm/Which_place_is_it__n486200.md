Question: Which place is it?

Reference Answer: street

Image path: ./sampled_GQA/n486200.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Which place is it?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Which place is it?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABCAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDwXFLinYoxQA3FKBTsU5VyeRQAifLkEZU9RTxHtVxnIK5B9a1J7KGGwgnBJMy5HHTkjn8jVNUIPl7SwbjAHOfah26Anco4pMVevtPudNumtryFoZlAJRhzgjIqqVoTvqhkWKTFSFaQigRHinmGQQ+cY3EZOA5HBP1pyKrSKrNtUkAt6D1rsrM3X9kNILHz7O1+ViYgy8dwM8/UVlUqclrIqMbnD8e1Fdb/AMJXaoSqafEB6bFoqfaz/kHyruc2EYqWAJA6n0oAqRWdBx6g/lTlRmV5NhKg/MwHAzW3MSRAVIgqVPKEMishLsRtfP3cdfzqe3sLieN5YIJ3RMBnSIsFJ6ZIpcy6jtc3b+3K29hEYlCmyjBAGOckk/XJpmlxRR6tayqobytzkZ4JCk4/StjxxEkOr2kMUSqq2cZIGRyc5NZel20nnQzkjaxkjCjt8h5/8ep+1i4qJSg9zO17UhrutXd/5RiilwVU8lAFAH8qxSnoRWzLp8yuYth2qecdzVSezeNsBGPHYUo8sVyoJLUzytMIqeRGUbmRgCSMkd6i5yAAcnpx1qiCMiu48Kma58NXdqspRsukb5+7kA/zqDXNGs7XwVpV5CgE7KrSP137xn9OKl8Alpo72FGwysjAYyDkEH+Qrnqy5qba6GqjyyszAvBqENy0d1GXlXjd5anI9c45or0SW3RnyygmiuBY7TWJt7DzOO8aaPp2meJZbbRLlprNwHjWUFXi3AHaxPXgjB/OmalpMukaZajerfa13kDr2/LtXtvif4Z+E4Zo7mSW9tw65kfzTKWOVVeGBrD/AOFc+GL+Jni8R3hiixzJFkLuOB1A6kY/Cuh4qnt28jKNN2ujxhYmdkQAFmYDn3OK3dNSWLUZ3tpWfa3zSqXGGweh6dePoK9Li+EuhsGlh8SriJhuZohhTnjPPqKsSeANLso/7QGqrfSSXGHaGQgFmz1UZAPB5p/WYPRDUWjmfGuj31zq1zqKW0jWNpFDHLP2Vmyf6/qKreG9LuJGtbu4tbltJM7Rs4XCucfMAfXA6+1eiXem3WtWmqabBdWyRSLHPcRyI29AuMEEAjHHSm6LpKyaVFoiajZTQW8zSEwlmkVnBHI2jjmquyorQzp/BenNd3eszxTxaFcLJ9k2E70fA2luvyk7q5XxL4TXSdK0aZY7sXd5EzSq8fy5z8uzAzyOx5r3WPwsW0uC1NyCsTB1bYQc/Q1n+JvDV14gurK6hubT7RYtyd5HRgefTpWklZ3sQpJ6XPALnww1tfWGn3089lNIS16k8B/0UcEP15XHfjpWTJYSWNob+C6iknhvDCkaj5ioGRKAeq9se9e5atomvz+LNT1m0gtLmHU7I2oiS9+VRtCk5OAenSuXu/h54lv/AAzpOiz6aGh06Z3aSK6iyVY8genTv6UXTJOf/s63votE0y6hlS1mt0Lx+YfkbByAD05AqLQ9PTSvF+q6bajZEiGSHJyQBtI/Q10F54evvD0eiLc2NwnkypBAXdGMhLcD5SfXHbrW5Hot3bapqGoXXh2eE3FqkTXTx/6sqSSc9gRtBI9MVzWmrp7f8E0vc5uZcSsM9DRU14E+1yhHBweSPWivJnpJo6o6o77xZLeXN5HaC180xiMvgkc5LDHI9PXt3rL07QNa8iRTbQLvaNsNvGQhJGPmPGTXkmn+ML7TrhriGR5JHTbIJWLhhnPfnNdDY/FjXYtr/Zon2LtwYieK7lhqiRz+0XQ9Kj0y+tFmS6tgkDskkpjZ2cBSTkfLjHJ61o6Np2lfZbuCJZrhY5HklFyoysoQsOMDn5wfxrgZPEusXcc2qtNKPPjMRtVJMa7h1x1/XAwKqWfibVV8QlVnlhinmJnVSdj5wpPPsAOvQVapSTu+5LbaPQLW5W8j1ews7NYL/wCwsFvVJLMx4GR3A3Z/CpbTS/7N1JBDc3E85dUnQKpVTuzxzkAk/wD6q5afxlNDfnSUlvDEAFATJUkgH06fQ0618RNId1vYaoDgPv8ALYFyAOMgcn0+lbylraw6cnF3R6x/a1payyWssrb4FXcdvXPTpVXS7zTr37c1lLJIN/73eMYyD04+tcTo+uapqGp3ROiXUbOm0PLCwZgvIJJx64/CotcuvEC2fkWGm3UbKWwYlABzznrxyTVVK700MlA6C2tNAuNKtbJZJzZee8SeZlW83cPl4HTJNXYrbQIrTVYlupPKO0XXJPl4Y47eufWvDbvSPFzv81perubLb5VVT+O6tNbnxFb2ItlsLSFJFAuMMA74PALBuRgfnUxnFa3RTg3segfEPSbS28KtewJKZtPntAGcdFjlU5H4Hr0rofEWuWC291proZZJ4mjUJhlyQcbsHI5rxHV9Q8QXljcw6jfTi0dWabNyW4x1I74wPyqjoqXsVqrOIsTS/aDub+9g8dfr+NaKrePui5GnqThvKZ0eN92efMbc2cDIJ9jx+FFJd3Sm6k3gIQxGFJOffpRXI6KbuzZTtoY5HOOxxmrtoBleBycUUVMi0dFYKAowAOfStkkm3dSSQVwR6iiisl8SG9maQkdLSGNXZU8tflBwOo7V2SSyNHbFnYnzx1P+yaKK6Y/xPmzGWyEuLu5WUgXEoG3oHNZ1xcTtGd00h47saKK7avwGUdznJv3k53/Nx35rFvo0BXCLyBniiiuPqdKMXVkT+zL07VyIWwcdODVTTlB0GwOBn7OnP4UUVvD4SH8Ql5zdyE+o/kKKKKxRLP/Z">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Which place is it?')=<b><span style='color: green;'>city</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>city</span></b></div><hr>

Answer: city


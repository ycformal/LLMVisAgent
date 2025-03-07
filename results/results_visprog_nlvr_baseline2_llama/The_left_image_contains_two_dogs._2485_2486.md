Question: The left image contains two dogs.

Reference Answer: False

Left image URL: https://d2gg9evh47fn9z.cloudfront.net/800px_COLOURBOX8819752.jpg

Right image URL: http://static3.bigstockphoto.com/thumbs/7/6/1/large1500/167459102.jpg

Original program:

```
ANSWER0=VQA(image=LEFT,question='How many dogs are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Program:

```
ANSWER0=VQA(image=LEFT,question='How many dogs are in the image?')
ANSWER1=EVAL(expr='{ANSWER0} == 2')
FINAL_ANSWER=RESULT(var=ANSWER1)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABkAEMDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAorifGXxGtvB2pW1nNp812Z4948iVdy845U9veuE1r4x+IpEb+zNKt7NCMKZsySD+Q/Sgdj3Givm2z+Knim0v0+0asZmY/6qWFdrewOOteu+DviRpfijZaSkWmp45gf7r/7h7/Tr9adgsdrRRRSEFFFFABUF7eW+n2U15dSLHBChd3Y4AAqeub8eQW0/g3URdyiOFYi5JPUjkD8TgUAfNev6p/wkXiTUNYluZ0V5tyIG+cxngAdhjj6Cta0voDpMmlT2Uv2sIJbedjvZhnnJwOlcvpDBPFVjJdbTayTqkqPyuxj1P06/hW/eWiabdXCQgN85i35JJVWz1PTOBnFDs3Y01SuUb94ruzks4LBWkjB8y7lbbtk6gLnjArJtbyRpFWUNFewkHcrYJ/2gR/St6zsbnWL63tYZjFkAOGY4Zc5Jz6nP5YrBvlso9UktLDcIraZxFcHh2ye4zjA5A+pzTTV7Ar7n098M/EEniHwbby3EjyXVuxgmeRtzMw6H8QRXY14v8CHuV/tSJwWhwh3BuN2T2/z0r2ikRJWYUUUUCCuc8e2yXfgbWInXcPs7MB7jkV0dUdYt3vdGvrWB1WaWB0QnnBIIFAI+OtUtJwVPlPGg+Tce1dRdyM0eZmDSNGCSP72OtYl/JMlvNCzELk53Hpz1/lXS6HoGq65of2q2njhgMnlBJlzuIABYcdOf0pebNlFy0RkC/udHs5buHh1BAfqV44x+JH5VgaNbveB9/JcHBP8TYzXX+J9E/snQbiKadyHkVFwv3gB+nOfyrl7R1t4YD8oxIrbh1HFCa1CSadj1T4Ma/aaReXtvqU8Fsk8alZZSRuYHGM9AMc816jq/wARvDGjq3makk8oGRFbfvCfxHH618ww3srSzeWm1B046/55pS/2iPzoh7Mnp9KaJcbnuU3x10dJWVNLu3UHhjIgz+GaK8CJOeBRTsTZH2nURjJQIQCB196lopEngfjf4VXdnqt/qdnJaRaI+ZiJHO6EnblAMc5I4H4U7wj4j0Kz0KLT7rULOyms3YOs7+X5gJJDj16449K2PjV4neJLfQLb7xImnP8A6CP5n8q8Dv0a5sZJ2UqFkwpI+8PWpaudFKo4anQ+NPGkXiG+Npa7EsI5yY3wcyqO5J6d8D3rmuSYx0CAlcHr7VmQwSO+FjJxkZrQmkMci2sIjJOFLBeR9KdrbEc3M22aMs8UMSKAY2YZABzj3p1k+AT91XbHHY1TSDauJTvc84J5HrWgQsUOOi8Z/wAaVy0h7WTyMWUpg+poq5HFuQMHIzzxRU87K5EfX9FFFaHKfMvxPku28c6t9sYFgQke3oE2jb+hrkUtEl0tt4JGCQvv2Ndf8SraWDxnqqzFyzOXUsMZUgEY9u34VjWenyLZRK8ZCgZOR2/yKzudCV0cwuniCGEsFJkVj8w+6yjP6ildYxF5oVd+3dn/AIEBW89g8v2a4IxGjN8vrnj+lZT2isSp4XYU/DJ/+tTuLlK6oPOaQjkn8qvRlXjGPvDkA+lUZW8uNXJwSBkf7VLFM8YjIbdgHr2NS9TRaGlJqHkuY/JBx/tUVTdZ5m8xWTBA7e1FKw7s+0KKKK2OM4Hxz4O0/VdSs9TnkWNywicM3+s4+UD6c/nUOqeF7e60tVhVcgDBIwcVo/ESOKPT9P1O43G2sLoSyBfcFQT7AkU7UrgLpzeVn/ViTOew5rmqrW50U5O1jn7fwDbR6SsN0uTz09zkVyet/Ca+MoksplKH+A8Feld7P41trfw8byRkaSPlkTlvyqDwj4/sfFFnLIT5csf3oz/Dntn/AD1qXpqUpM8nvPBEmmWcsl1DvdG+YEVxFzEwvH8piFVgPm6H1r3bxf4hhS3aCVYg7AKQ/oc7T9MjH414hfS/6Q++MR7mPH90irhLmQ2rDEt1VANxHsKKnS4j2coCeeQaKCz7LoooroOEy/Eih/DWpKwBDW7jkA9R715P4Wv7q6lv7G5maaG3cW0e/rsAH6miisquxrTOE8ZL5Hj/AMmJmSJlUFAcA5B7Vp+CJ306x1D7OFGJs5Iz2oorGXwm0dza8dTfadJjleOMPvVSQuMjDHmvHdRkYux75/oKKKunsDIIyQgAJ/OiiitBH//Z">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='How many dogs are in the image?')=<b><span style='color: green;'>1</span></b></div><hr><div><b><span style='color: blue;'>ANSWER1</span></b>=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="ANSWER0 == 2")=<b><span style='color: red;'>EVAL</span></b>(<b><span style='color: darkorange;'>expression</span></b>="1 == 2")=<b><span style='color: green;'>False</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER1</span></b> -> <b><span style='color: green;'>False</span></b></div><hr>

Answer: False


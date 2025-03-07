Question: The right image contains exactly two penguins facing each other.

Reference Answer: False

Left image URL: https://birdingthebrookeandbeyond.files.wordpress.com/2017/03/three-kings-at-volunteer-point.jpg?w=640

Right image URL: https://static.wixstatic.com/media/67788f_1f74d118ecac4826b69e9f3ec61ef366.jpg_srz_316_357_85_22_0.50_1.20_0.00_jpg_srz

Original program:

```
The statement is True.
```
Program:

```
ANSWER0=VQA(image=IMAGE,question="Is 'The right image contains exactly two penguins facing each other.' true or false?")
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAvAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDhI7x94VFDEDk7lHFWI57qQr+7AXpu88YwfamQWUEgaVph7gMT+dakNrF5XCFlJ7dPz61yNxXQCG3W6Ee0zxM+fuiQt+lLHbobppHiIduCQSOff1Ht71rw6fbLGzlAoBPJU5Pr9abczrBbO1vZySNHCZcIQAcAkLxkg/QZqLtvQLvYVbd5Aqx3TKm3jKcfTJqo95ZWmqLYSzTNeSAFFRT0+v8AhUOiarLqunT396Fgjh3/ALojqAMk8+nT610Hg7QoPE1/d6hc2ZW6sLPzfOBOcuG2IB06Akn6etVGjJ8z7BdIqRoTHukjWI5+6XLf99YpTHA7syA8dTtwB16evSru1GcKWfGcggdPp71VkOyHekRYA/dboV/nWaEQRxx5G2MNzkFcelSNpWparHcxaZJbJcrCZV8wgAY6/Xv27VFHJE0bhlAKkgkgZx9RyKr65oJ0u907UbaIpeXduAVHQpISF68hiAffmtaPPOXIh8yS1HGzuIXigkvLfztqpKyx7laTHO1gBgZyaoWllemAvq1uY53L7YkUqcA4zz1Brp/A01tqniGBXtwIvNZ8OM4IG7AP1rP8TTBPE18gllWdL6dNrNwFPzrj2xXWqcm5dlcn2m2mpll1TCtbyRkD7uB/jRTvMkXhpsHqfkBorK7LuUoBEqkRwbTjB2Dv/OtC1d4wd8MgUnrzyfy5rIiF07eZITgdBCFGR2yxrSiurhkb9wyqBnBIx/8Ar4rmehPzHXXhu88Y6tZWel3kcMqLI/kTFgX2kbm9OMiuysPC15pvjax0WUbIAv2uWWI48yFRgj2y5UHv19c1j/B/UbeXxbIbiSNZ1sGSFCMFyXDNg+uBXrmoypLqtrcRxjzUgliLdypKHH5ivQhh7STWyE5XVjmvijb2cXw81J7a2hiMYD7o1wRlgG6fWsf4NXUQ0zXmaQtcebGXBPHl7Dtx/wCPVseNILjUfA+tQqjZNuTgdwCCf5VwXwmuvsWl6te3D+VC8KQq5/icBuB6nkVahCn7sVZCbb1Y7zbiF1k+/GcuAFzwfr/niqjX1sP9d5RXJUhhtJ/MY9KmNpFMWeQSK4PTeeBx27VWuY4jEQqFnHBBYduMkgfzryYu7KuhTGksLSR4VHGVXGRj6itn4m3LWv2G9hmCrNZxDYP4TG3B/wDHv0rmZJHhdYhJsULjcen1HI/WrXiyUat4Msr4fMbIG1lPqxKlTj3AP5V24GdpuLMp62E8CXE9hrJkbhfNDNnPAOR/hWx8UwUvNH1QQBfOmEbyleGwMDP4GuOju76ea0nM/kRx20MTqGyX2DAyD0rovG2r/wBqeCNCD3CKYbotKWbnCDGf1zXRCcXKcV/XQTTumUfmPRYW99poqFrOCc+Y8jZYZBKZyPUUV53Mu5tZlOJYoykhLu+duMHA44PHBrSi2uULFpM9dy9R9PWueiuEVcthNxwAq1bFxcPAqxyna+AinjjOKbhcTZsafpMa+I/BcNkmy5aRzcunBYJMSGP/AADivYdfu7bw9b2uq6hKy2oZ0YqM9QCP/QTXkGg69caSrMbGGe7iVlt5WY/ITjcD32nCnA7/AFrP1TU9a165dtavBLnAWFRiNB22r0/HvXf7eMY76knp2q/EfStU0vU9L0qK4+0NbBVlKhdjOD1B9B/OvOra4v8ATrKWzt7iKOydxIyFMt5nTduz+lULeC3jlGFRZC2wELyT9e3SrsIkZ2R1ZU9cjLVy1a0mVYR7hrjC3TSMQxYlFK9uvH9am8ySNmnVsggZXoW6dRTWni2mcExoBkZA7ccYqrLeZy4lXOCfmjyTXPuOzMrW/EFvbXoSZZZSYwR8uAOSMe9ZJ8UubaW0WeZLaVlZl8pSQRnBB+hNVfFrl9ViJIOIFAIUDPJ7Vg11U4JK63CyOmi1jS3Ym6N1Ju6goCAPzq1DrugWu4Q2bjd1JiB/ma4+inKkpbthY7pfG1oihRBIQOmQBRXC0VH1eAz/2Q==">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is 'The right image contains exactly two penguins facing each other.' true or false?')=<b><span style='color: green;'>true</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>true</span></b></div><hr>

Answer: true


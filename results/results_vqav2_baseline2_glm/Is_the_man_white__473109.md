Question: Is the man white?

Reference Answer: no

Image path: ./sampled_GQA/473109.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Is the man white?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Is the man white?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDqM4pc1G4ZkIVirHuKIwyoA7bj614J6I7aytuT8V7Gp42DjI/EHtUY5oYhfmH3h3H9aqJLJ5JUgTc5/D1qhJrSoCQmOccmq9/cYUl2AJGOtc3POcjBBPStYJyNYU01qdaNatY5Sk6urf3xyKvLJHPHvjdXQ9wa88llY/KSfwNWdI1prK7C7i0R4ZM9vatuRpFOimrxOzPyNtPQ9DUUgq18k0aspyjDINRbGOVwSR6VhKRgmZ8wwM1m3Dhep/Ct97Mup3HH0rmVJBZXH7xThjWcWma03crneTkJx7mipjnNFaG9zpclfvcj1FLvUYOevTHemh2f7vA/vEU5Igpyp+b1PeoscbZIis33vlHoOtJdEJCFA6njFSI3OGGDUWoOIrcSN0U1VtCU9Tm9Ut5Jud+COmTgVlLDLAjH7RHz2wTVDWPElxHcvGrxMQ2Cu3PPpms2HV7y6fbEDnviuqFKfLd7G6qRvymxPAGj5nyf93GTVBYzE33iTWfJql0rlG5kz3qWG/kdVzKHzyQVxkd8HvXQoSSGq0LnpXgu4N1YzW8hY+SwI+h7fmP1rpygAwBgVy/gCItBeSj7rbAP1rsTHXm1V7zOarJc7sUGQ1y+pWrRX8jKOp5HqDXaGL2rD1m32zo+PvL/ACrFaMqlLUxRBgDNFXxEGAbGciitDfmJkkDdDz6VMprPRgRzVmN27cj3q0YSRdXnrWfqdus8satIwiKEEZ4zkHP1q4hz941V1QhTBuOFYMv48Gm9jOLtI4TVtNJuD9hhUwRtuyVzhu7Z9TWp4S0WJYZb5yuDlFJ7nvUOsXklxKbOBtoPAArG1O61O0mW3MkK28KBIlU4wB3471uuacOS51JcvvMfqFj5GqN5ZUSknB9f8iqJtwsSQSBUVMlEAxtJ60lnLdXcrRTyRbM7gwPIParU86yPtf7wHX3rdc0bRYTcXeaO1+HNq731zLuPlwwhNueCzHrj6A16J5Wa4X4VxyyR6lOVPlOUCkjjjNekrD7Vx1Y+8zz6lTmkURBntWN4phkh0d7iFQZIskAj2/8ArV1Xlgdqp6rbCfS7hMZ+TP5c1hy6mlOdpJs4LRrlrvSoZ7lRDIwztxwR2I9iKKdaW4sIBBEWaMEld5ztHoPYUVbavodU5JybjsU4mIOT81XYpAcVjxTE9s+9dLo+h32pqHVDHD/z0kBAP07mrSIqtLViRkVW120mutHl+z7hNF+8QDknHUflmuvt/CcagF7xj6hUA/rWlbaVZWcyuN7Mnzb3bgfhW0abuccqi6Hzu99vtHnj+W5Y7Bjt7ipUs5bizDzX9qjY4QgsT9T610PxE8I/2RqcuoaaAbOYl2UH/Vseo+ledXRVlHmM+/HOOK7FRvtoawxKWstTR1C1MOPKvIpX9VXFUmlmnEcagtNIdgC9Sc4qqrJHGTHuLeua734X+H2utet9Xu4t1rbOGTcPvP2IHt1rVQ5VrqY1a3O/d0PXvCPhx/D3he0tJj+/Vd8oHZ25I/Dp+Fbe2rIkSVCUcMPaoa86om3qZDMUx0DKVPQjBqU0xqxaNYs8/miMczoRypIoq9rMfk6pMOzEMPxopWR1rVG5pXhLTNHj81k+1TqM+ZMBgfRegrYS5SWE4+U7eAfpTy4dGPbbWJA7Rucfwtg8ds4ruo0k077o4m3LVmraM/lsxYn0FNuVKzwgn5XBQ/XqKitpRtEe4gg8+/BqzejfasV6qNwrbaZnJanmvxStm/s+2hCZidyxYA4UgY5bB2jBOeD7V4lcW0yYUOSCNw4zkYzn8vxA64r3T4iSw3fhlGmETSQSht0kbMu3BBPy854x+NeSQW1xq1yFjibYzKXlJ3bs8s2cDLE456gAA5FdsE7e8PdJGLaJJHNukiWZUGfLZioJxkZ79+g610qeJNRmgSF70w2zjCxxfu0XH8J2jIx3IzgYGOavD4f3MtjJcQXLGaMb2iKld/XoRznjp6iudSGaIqTFMxJXy2ijK8kKSMn7vLAggE59M1fLG4ltofQ/hLUGu/D9hOWYySQjeSACSODkDgdM/jXR70ZtvfuRXIeFbFtJ8PWdo+0vHHhygONxOTj8Sa6NJEK/K2G75rjqwT1FrctsMVExphnBYAZGKYkvmytHwGFcVSi0ro2iUL/Tlu7gSHGQuKKtl/mI9DiisbI2UmhILxYkO84Gf4Rkf/rrMkmVbkjqHJGDz+f4Vy8+sybYlBVUZPmBOAexIPrmo7bWpXlcq4UCIkqnzYZTgnPrivfhg3G7B0mmdvZy/wClH0G3nH4Vrq4kjBwQCAOR1rkrC5EiW8xBG+NeZDk9TXR2kuYVOR90cgn3rjrws7mE42MyeBJYHt3VXVQQVIGCM+9ZTaXbKpQQKueO3Hat6FY43muJmGN5AGc55IomKXCq6BQeDgf59q1U7PYizOe0+ze2EiqQcnAKnBx7/nXPJ4MiHiNr7aoUvvEexRhz1OR/L1Oa7qO3L7iu7aGPp6Zplyq26eYdoAGOo/p+FaOopOwJW2E3iJQgHQY6U5JCT8pwfxFUYQ8p8wgAE9wRV+Jdo45PtmiVkOxZBbAY84PNVZ5mhuI5FY9Rz2/zirYdfLIY4+v/ANesvUJwssKHGDk59CKygruxcUXkuDMXk8towzHAYjP6UVUWUooABPf6UVxSw07uy0NLo86vwEW9RVASO4/dr/c47enrUmg5l+2FmICo4CgDHYGiivqfsM657I6vSPkt7JV4G0L+AxXWJlIFOSfk79uDRRXiYrc46u5nXssgYLvON+cYHtU0Lvs+8eMfzooosuVEMgsbqUmXJBOAeR71HeSPOqxuxK7lOBx60UVdl7Ri6j4lCwL1PHc+3/1qNx8tm7jpyaKKXUY5JmdeijHoPeszVWKG2YcnceTz2oopw+IqO5MCxyNxGOODRRRWgj//2Q==">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Is the man white?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: no


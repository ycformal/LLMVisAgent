Question: Do these animals produce bacon?

Reference Answer: no

Image path: ./sampled_GQA/414679.jpg

Original program:

```
ANSWER0=VQA(image=IMAGE,question='Do these animals produce bacon?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Program:

```
ANSWER0=VQA(image=IMAGE,question='Do these animals produce bacon?')
FINAL_RESULT=RESULT(var=ANSWER0)
```
Rationale:

<hr><div><b><span style='color: blue;'>ANSWER0</span></b>=<b><span style='color: red;'>VQA</span></b>(<b><span style='color: darkorange;'>image</span></b>=<img style="vertical-align:middle" src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABDAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDvC8b9iPpSoE3DDZxyc1k3N39ms5pl5KIWA9TVnwfq7J/od+VuknJkEpUDbwS3Hpx9a0rV40pKD6ipUJVYua6F+aQvJuGOfSmhmxWLLrkusa239jwImnRf62UIPlHY+5PpW3bX0EmI5bOYE8gkZ/UVksdS2saPBVN2A3eopxZRjlvenzNaxrnY6E9AXqg1zbterbm9S3d1DIkgA3gnHyk9ee1V9cp3siVhJ7sttIG6DH0pyExkMBz6kdKmuNCvlhYQXMAlK5XzEJAPvjtWGJr7T9Rgi1C4sHEjmKWOFt0iEgbNoJ5ye/br2q/bxsR7F30NXzSzFgQ5/wBqopLjPzBduOuDXM6qdS06/IbUA6MS0RQcMPQ9qybrWtakztmhiXHO0Dj8TXRHlfU55Sa3R3a30m0gY+veozdO3y5zXmZ8R6nbsQt6ZOedwBFaFt4wl8rFw6Fv9hSD+NX7MlVTu2ll3c7f++aK4Q+LLdjnbP8A985/rRT9n5h7QhtvEbPLFb3E7bZWEbAqrcHg9h61ofCzVY31y50+7t4zJFGfKndju25A2gdPqa47S4N1/bbg0bNKoQsvUgjqT0FdPomjrZ60+qeZuuG81bS2DANI5BGc54UDJAPJI9q8/F8rlFrdHZhHJQknszcttQjme7TT0WOy+1v5axqApCnGBj3yTn1rqkkVIBjBc9T6V5Lplx/ZOoyQWsMW7zsMXl5ZwMEYHHc8+orsRrYeXyxJhkPzKeuBXlVI8ktep6lOXtI6dDdv5ZUsJpITH5wU+X5nQt2zXNWb+IF1bT5NQlSeyMqPMqxBioz90ZHB6dOtR3muPLNHaxNuaRgiqOpJNW/EOuQabbESBjsYFNp9OlQpO6aLcdLM9NW8glthd+Yvkt0YntXkWs65FdeIdXk0a7to/tCAMZQA0uByEyMkH2616LqWuaf/AGC5a7gikMasBKMKpKggN+deB3twbie/yABHcfJgDP8AdOfb5a9NxdT3UzzFL2XvSR6bZWlvc+Brw6nJFEbR90EyKDtbAzxnB9CB/MV50b67unaNZYee7Dj8KzzfyyW6W0srNDGxdFzwpPUioDiRyqjDddo9K7KMHBWbOOtUVR8yVi3ci4tmw6Bh6ociq3n5IBVwPTbUXmywuGDFWHSmLcyFvnQsevHauhSdjnaVyyJ3GdquAewFFQNdS5/1TUUXYWOkgvsgvPEpVFwuFxg+xNdTZxWr2HhyyR44XkZ9QurkEZ+UMP5dK4qO3KYEglZh1znn9f1ror/VfsXh3w/cwwNugeSN+enP3SfpmvCi9Xc9fWJg3bQWt7LCkG1BKzh4+hBOeMfWk1yWYXcctnJ/rolZCv8AEcd61fHAisHtNVsg89hqMKyIVGcELk5PTOMH8DUesOp+GOleIILVFmVhETk/MCWAJ/Km4OTdwhNx2ZiaDc6la+IrO5lV3Vm2bFUs2WG3j8667W7eyuAb26J2xSeWkeeC3YknsP6159pfia5k1O0QqFbzkAdWIx8w5r03xd5T/D221NFVicSSFR1/hJ/MZpKk3q0Wq7sYniWzvm03SdTu9QmmaV/JVUUeVF3Xjv07+lUrfSm8iR2nyZASwA53Hrj86s3viGC70m20u1sRbmFhcXR3b0c7cKy55B+9xXOL4vtojtEM6qpxgMOKuXtIL92YzalubEvh+NFQo3zAZYgE5/CmvpkQQCW9VQMkALz+f1qg3jezyWKylGXDAL+XGamtfGWmSq0codTj5cqMfjU+1xG7I9nAlbQQCXkukbPIIHU55ofRQFZFm4fBKk4INSpqdo7KAy5IyHfJDU8P5oXy9hIblgQCP8/1pfWa3Vj9lDsQf8I/EeWuAD9QKKebntISWHt/hRR9Zq9w9nT7F1ZZSmBsO4fLlfSmq09xF9nmWLEmQ6FcqfTHPrR5U3yhZHXjHKZzVoWrOisJJCV6jGDWeqOpKRlXV5d2+mpp8iRpb5+SENgKTjP0Paux0i007VPh3caOVjzHE5S3kY8MHynX1Nc9/ZkrMWeNS68B2OTSi2ntyyqQPMPOGxuxzz+PP4VcZtCcG9zzeYyRzK0disZjIbERYqpH1rqdXh1C4+GelatNPOENzNbeQCQjJnIbA99wrbmETWwj+zxMOhTBx+Q6mtjWbixb4d21klt5kqlgkfmbI1YNn5h1GQ3HrWkal9GjN02tjivBkc/iy/OhlY4ZVsJFjdRgyYYNhj3PJ57VzP2fTrK/mt7+O5ikhlZHXIbBBxg9PSux8ORSeH/EcOqoP3sGd0MS5BUjDLnscd6qeN4bC71R9VgItzcAMbZYXA3dzuPc/rVKcXoQ4talGDSdM1RAkFzDhjkfu9rD2Iq3N4QtAAgmVCRjce59T+FcmzEE7QR6c4q5p9/PApVpbh2P3QnP6Zo5X0Y1KL3R1WmaC1tGY0v0nz0AHT8avrps8e5wcyDow+X+VZunalLKpiAlaRvmLOmC3v6VaF1f25d/MaUKeVYZP5Y71g02y1y9ETG0djkKz+/lZoqKK+ubiMSfvISeqMOQfzopcrHyrsQ2F5O9yUaQ7cjjArVivJ1gciTnrnA9aKKgi7LhuJTcwKX4JOeKsNI2Bz/F3/GiiqOhPQOJYA0gDFsk5H0qKBVjlaNFUIBwNooooKQ4OxZhngDjFRbQsfmAfMw5NFFMDitcijOrzHYuSQenfFdFo+n2kel2kywIJXTLP3JoorafwoyiveZpxHO1TjBPpRMitEGIGR3oorHqWUm4P4CiiiqA/9k=">,&nbsp;<b><span style='color: darkorange;'>question</span></b>='Do these animals produce bacon?')=<b><span style='color: green;'>no</span></b></div><hr><div><b><span style='color: red;'>RESULT</span></b> -> <b><span style='color: blue;'>ANSWER0</span></b> -> <b><span style='color: green;'>no</span></b></div><hr>

Answer: No


# Hate-Speech-Server

Creating a hate speech server for a mobile platform. This utilizes twitter data to detect hate speech on a mobile coding forum. 

Mobile application can be found at https://github.com/sashank007/Mobile-Hate-Speech-Detection-Forum .

# Instructions to Run

1. make sure python 3.x is installed
2. do a pip install flask(please refer to this link for flask help : https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
3. do a pip install flask-sqlalchemy and other flask db required packages
4. make sure FLASK_APP environment variable is set to app.py if not running
5. do a flask run to make sure server is running
6. download ngrok https://ngrok.com/download and do ./ngrok http 5000
7. copy the url retrieved into the MainActivity.class static url -> nograkUrl
8. rebuild the apk
9. you can now run it.. have fun!

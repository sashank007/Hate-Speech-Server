from app import db

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True)
    time = db.Column(db.Integer , index=True)
    post=db.Column(db.String(140))
    body  = db.Column(db.String(140))

    def __repr__(self):
        return '<Post {} {} >'.format(self.username , self.post)    
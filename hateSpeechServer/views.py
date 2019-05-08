from app import app
from flask import g, render_template, jsonify,request
from app.models import Restaurant,City,Fifa
import logging
from app.search import query_index
from app.forms import SearchForm
toto_logger = logging.getLogger("toto")
toto_logger.setLevel(logging.DEBUG);
console_handler = logging.StreamHandler()
toto_logger.addHandler(console_handler)

@app.route('/')
@app.route('/home' , methods=['GET' , 'POST'])
def home():
    toto_logger.debug("request coming in " +str(request));
    if request.method=='POST':
        val = request.get_json();
        return jsonify({'your data' : val})
    elif request.method=='GET':
        return render_template('home.html')


@app.route('/search/' , methods=['GET'])
def search():
    form=SearchForm()
    _index = 'fifa'
    _doc_type='fifa_players'
    _search_field='Name'
    searchRequest=""
    # searchReq = request.form['search']
    query_string = request.query_string
    toto_logger.debug("current search request:" + request.query_string);
    if query_string!="":
        searchRequest = query_string.split('search=')[1]
    # data, total = Fifa.search(request.query_string,"Name")
    players , total = query_index(_index,_doc_type , searchRequest,_search_field)
    toto_logger.debug("players " + str(players));
    return render_template('search.html', title='Fifa Player Search',
                           data=players,form=form)

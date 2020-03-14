
# coding: utf-8

# In[5]:

from flask import Flask , request
from flask_restplus import Api , Resource , fields
from flask_cors import CORS, cross_origin
#import ChromeExtensionAPI_controller as controller



# In[6]:


import Controller as controller


# In[2]:

flask_app = Flask(__name__)
CORS(flask_app)
app = Api(app = flask_app,
		  version = "1.0",
		  title = "Sayak Extension Server",
		  description = "")

#app.config['CORS_HEADERS'] = 'Content-Type'
name_space = app.namespace('sayakext', description='Manage names')



#cors = CORS(app)
@name_space.route("/inner/",methods =['POST'])
#@cross_origin()
class Inner(Resource):
	#@app.expect(model)
    def post(self):
        try:
            data_req = request.get_json()
            text=data_req['query']
            results = controller.GetResults(text)
            #results = kwg.get_all_resultsOf_query_innerText(query)
            return results
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not save information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not save information", statusCode="400")
@name_space.route("/outer/",methods =['POST'])
#@cross_origin()
class Outer(Resource):
	#@app.expect(query)
    def post(self):
        try:
            data_req = request.get_json()
            text=data_req['query']
            #print(text)
            results = controller.GetResults(text)
            return results
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not save information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not save information", statusCode="400")
@name_space.route("/getCompareSitesummary/",methods =['POST'])
#@cross_origin()
class compare(Resource):
	#@app.expect(query)
    def post(self):
        try:
            data_req = request.get_json()
            text=data_req['query']
            #print(text)
            results = controller.compare_site(text)
            return results
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not save information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not save information", statusCode="400")

@name_space.route("/changesliderrequest/",methods =['POST'])
#@cross_origin()
class changesliderrequest(Resource):
	#@app.expect(query)
    def post(self):
        try:
            data_req = request.get_json()
            text=data_req['query'];
            #print(text)
            selection=data_req['selection'];
            #print(selection)
            slidervalue=data_req['slidervalue'];
            #print(slidervalue)
            
            #print(text)
            results = controller.slider_response(text,selection,slidervalue)
            return results
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not save information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not save information", statusCode="400")


# In[ ]:




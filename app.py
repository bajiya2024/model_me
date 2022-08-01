from flask import Flask, jsonify, request
from HelperClass import model_data_maker
app = Flask(__name__)


@app.route('/api/v1.0/')
def home():
    res = {}
    res['msg'] = 'Working'
    return jsonify(res)


@app.route('/api/v1.0/predict_price', methods=["POST"])
def price_predict():
    if request.method == 'POST':
        params = request.json
        fields = ['Brand', 'Model_Info', 'Additional_Description', 'Locality', 'City', 'State']

        # validate input params
        inputs = {}
        for f in fields:
            if f in params and params.get(f):
                inputs[f] = params.get(f)
            else:
                return ("Please provide param : %s" % f, inputs)
        return model_data_maker.pre_process_data(inputs)


if __name__ == '__main__':
    app.run(host='localhost', port=2021,debug=True)
import traceback
import advertools as adv
import pandas as pd
import ernie
from ernie import SentenceClassifier
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/nlp_return/predict', methods=['POST'])
def predict():
    try:
        dataset = request.json

        data = pd.DataFrame(dataset)
        data = data.filter(['content'], axis=1)
        chinese_text = data['content'].to_list()
        predictions = classifier.predict(chinese_text)

        outputs = []
        for prediction in predictions:
            if prediction[0] > 0.5:
                outputs.append("False")
            else:
                outputs.append("True")

        data['has_negtv'] = outputs
        data = data.to_json(force_ascii=False)

        return data
    except Exception as e:

        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


@app.route('/nlp_return/word_freq', methods=['POST'])
def word_freq():
    try:
        dataset = request.json
        # filtering
        data = pd.DataFrame(dataset)
        data = data.filter(['content', 'has_negtv'], axis=1)

        # negative
        negative_comments_dataset = data.query("has_negtv == True")
        neg_text_list = list(negative_comments_dataset["content"])
        neg = adv.word_frequency(neg_text_list)[:20]

        # positive
        positive_comments_dataset = data.query("has_negtv == False")
        pos_text_list = list(positive_comments_dataset["content"])
        pos = adv.word_frequency(pos_text_list)[:20]

        freq = pd.DataFrame()
        freq["Negative"] = neg["word"]
        freq["Positive"] = pos["word"]
        freq = freq.to_json(force_ascii=False)

        return freq
    except Exception as e:

        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    classifier = SentenceClassifier(model_path='./sen_analysis')
    app.run(host='0.0.0.0',port=5001)

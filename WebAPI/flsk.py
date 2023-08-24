import joblib
import pandas as pd
import math
import re
from flask import Flask, request, jsonify
from collections import Counter

# code from flask official website - https://flask.palletsprojects.com/en/2.3.x/quickstart/#a-minimal-application
app = Flask(__name__)

# Load your machine learning model here
filename = 'RF.jobl'
modelLoad = joblib.load(filename)

# Setup the web API
@app.route('/predict', methods=['GET','POST'])
# Start with the prediction function
def predict():
    datab = request.data.decode('utf-8')  # Get the plain text from the request
    # Codes adapted from https://github.com/MariaZork/my-machine-learning-tutorials/blob/master/JS_obfuscaton_detection.ipynb for feature selection
    df = pd.DataFrame(data=[datab], columns=['js'])

    # removing empty scripts by checking the scripts column
    df = df[df['js'] != '']

    # Featurization
    df['js_length'] = df.js.apply(lambda x: len(x))
    df['num_spaces'] = df.js.apply(lambda x: x.count(' '))
    df['num_parenthesis'] = df.js.apply(lambda x: (x.count('(') + x.count(')')))
    df['num_slash'] = df.js.apply(lambda x: x.count('/'))
    df['num_plus'] = df.js.apply(lambda x: x.count('+'))
    df['num_point'] = df.js.apply(lambda x: x.count('.'))
    df['num_comma'] = df.js.apply(lambda x: x.count(','))
    df['num_semicolon'] = df.js.apply(lambda x: x.count(';'))
    df['num_alpha'] = df.js.apply(lambda x: len(re.findall(re.compile(r"\w"), x)))
    df['num_numeric'] = df.js.apply(lambda x: len(re.findall(re.compile(r"[0-9]"), x)))

    df['ratio_spaces'] = df['num_spaces'] / df['js_length']
    df['ratio_alpha'] = df['num_alpha'] / df['js_length']
    df['ratio_numeric'] = df['num_numeric'] / df['js_length']
    df['ratio_parenthesis'] = df['num_parenthesis'] / df['js_length']
    df['ratio_slash'] = df['num_slash'] / df['js_length']
    df['ratio_plus'] = df['num_plus'] / df['js_length']
    df['ratio_point'] = df['num_point'] / df['js_length']
    df['ratio_comma'] = df['num_comma'] / df['js_length']
    df['ratio_semicolon'] = df['num_semicolon'] / df['js_length']

    def entropy(s):
        p, lns = Counter(s), float(len(s))
        return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

    df['entropy'] = df.js.apply(lambda x: entropy(x))

    # String Operation: substring(), charAt(), split(), concat(), slice(), substr()

    df['num_string_oper'] = df.js.apply(lambda x: x.count('substring') +
                                                  x.count('charAt') +
                                                  x.count('split') +
                                                  x.count('concat') +
                                                  x.count('slice') +
                                                  x.count('substr'))

    df['r_num_string_oper'] = df['num_string_oper'] / df['js_length']

    # Encoding Operation: escape(), unescape(), string(), fromCharCode()

    df['num_encoding_oper'] = df.js.apply(lambda x: x.count('escape') +
                                                    x.count('unescape') +
                                                    x.count('string') +
                                                    x.count('fromCharCode'))

    df['r_num_encoding_oper'] = df['num_encoding_oper'] / df['js_length']

    # URL Redirection: setTimeout(), location.reload(), location.replace(), document.URL(), document.location(), document.referrer()

    df['num_url_redirection'] = df.js.apply(lambda x: x.count('setTimeout') +
                                                      x.count('location.reload') +
                                                      x.count('location.replace') +
                                                      x.count('document.URL') +
                                                      x.count('document.location') +
                                                      x.count('document.referrer'))

    df['r_num_url_redirection'] = df['num_url_redirection'] / df['js_length']

    # Specific Behaviors: eval(), setTime(), setInterval(), ActiveXObject(), createElement(), document.write(), document.writeln(), document.replaceChildren()

    df['num_specific_func'] = df.js.apply(lambda x: x.count('eval') +
                                                    x.count('setTime') +
                                                    x.count('setInterval') +
                                                    x.count('ActiveXObject') +
                                                    x.count('createElement') +
                                                    x.count('document.write') +
                                                    x.count('document.writeln') +
                                                    x.count('document.replaceChildren') +
                                                    x.count('window.execScript'))  # this item not in original code

    df['r_num_specific_func'] = df['num_specific_func'] / df['js_length']
    script =df.iloc[:, 1:]
    if df.empty:
        return jsonify({"result":  "no"})  # Return the result as JSON
    else:

        # need a sub routine that would check if df does not have any rows, no need to process just return 0

        result = modelLoad.predict(script)
        response = "yes" if result[0] == 1 else "no"

        print(response)
       # return result_list
        return jsonify({"result": response})  # Return the result as JSON


if __name__ == '__main__':
    app.run(debug=True)